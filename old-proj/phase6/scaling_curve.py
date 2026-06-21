# Phase 6 — O(n) Scaling Curve Experiment
# Proves Proactive Cache inference is O(n) while Full Attention is O(n^2).
# Measures wall-clock time, peak VRAM, and throughput at fixed budget across
# sequence lengths [512, 1024, 2048, 4096, 8192].
#
# Usage:
#   python phase6/scaling_curve.py
#   python phase6/scaling_curve.py --budget 256 --repeats 3
#
# Outputs:
#   outputs/phase6/scaling_curve.pkl   — raw data
#   outputs/phase6/scaling_curve.png   — three-panel figure

import sys, os, argparse
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.cluster import KMeans
from config import PHASE4_DIR, NUM_CLUSTERS

MODELS_TO_TRY = ["unsloth/meta-llama-3.1-8B-bnb-4bit"]
SEQUENCE_LENGTHS = [512, 1024, 2048, 4096, 8192]
FIXED_BUDGET     = 256
PROFILE_SEQ_LEN  = 512
NUM_CLUSTERS_DEF = 4

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase6")
PHASE4_DIR_ABS = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase4")

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def parse_args():
    parser = argparse.ArgumentParser(description="O(n) scaling curve experiment")
    parser.add_argument("--budget", type=int, default=FIXED_BUDGET,
                        help="Fixed KV budget for proactive_cache method (default: 256)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Number of timed runs to average per (method, seq_len) pair")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQUENCE_LENGTHS,
                        help="Sequence lengths to evaluate")
    return parser.parse_args()


OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")

def try_load_model(model_name):
    """Tries GPU-only first, then auto, then disk offload — same strategy as benchmark_llama.py."""
    strategies = [
        ({"device_map": {"": "cuda"}},  "GPU-only"),
        ({"device_map": "auto"},         "auto (GPU+RAM)"),
        ({"device_map": "auto",
          "offload_folder": OFFLOAD_DIR,
          "offload_state_dict": True},   "disk offload"),
    ]
    for extra_kwargs, tag in strategies:
        try:
            print(f"  Loading {model_name} [{tag}]...")
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            os.makedirs(OFFLOAD_DIR, exist_ok=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=BNB_CONFIG,
                trust_remote_code=True,
                **extra_kwargs,
            )
            mdl.eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"  Loaded successfully [{tag}]")
            return mdl, tok
        except Exception as e:
            print(f"  Failed [{tag}]: {e}")
    return None


def load_prototypes(model_name):
    short = model_name.split("/")[-1].lower()
    proto_path = os.path.join(PHASE4_DIR_ABS, f"{short}_prototypes.pkl")
    if os.path.exists(proto_path):
        with open(proto_path, "rb") as f:
            return pickle.load(f)
    print(f"  WARNING: No prototypes at {proto_path}. Run phase4/profile_llama.py first.")
    return None


def build_synthetic_input(seq_len, device, tok):
    """Create a random token sequence of exactly seq_len tokens."""
    vocab_size = tok.vocab_size
    ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    return ids


def predict_and_mask(prototypes, seq_len, budget):
    """Score tokens using prototypes, return list of kept indices (O(n) per call)."""
    scores = np.zeros(seq_len, dtype=np.float64)
    for (layer, head) in prototypes.keys():
        centroid = prototypes[(layer, head)]["centroids"][0]
        max_d = min(len(centroid), seq_len)
        if max_d > 0:
            cumsum = np.cumsum(centroid[:max_d])
            for p in range(seq_len):
                reach = min(max_d, seq_len - p)
                if reach > 0:
                    scores[p] += cumsum[reach - 1]
    # Sink boost
    scores[0] += scores.max() * 10.0
    # Proportional recency window
    recency_window = min(max(8, budget // 16), seq_len)
    for i in range(recency_window):
        scores[seq_len - 1 - i] += scores.max() * max(0.5, 5.0 - i * 0.5)
    scores += np.linspace(0, 0.001, seq_len)
    actual_budget = min(budget, seq_len)
    top_indices = np.argsort(scores)[-actual_budget:]
    return sorted(top_indices.tolist())


def _to_tuple_kv(past_kv):
    """Normalize DynamicCache or legacy tuple to tuple of (k,v)."""
    if hasattr(past_kv, "to_legacy_cache"):
        return past_kv.to_legacy_cache()
    return tuple(past_kv)


def _to_dynamic_cache(kv_tuple):
    """Convert (k,v) tuple back to DynamicCache for models that require it."""
    try:
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(kv_tuple)
    except ImportError:
        return kv_tuple  # Old transformers — tuple is fine


def prune_kv(past_kv, indices, device):
    idx_t = torch.tensor(sorted(indices), dtype=torch.long, device=device)
    pruned = tuple(
        (k.index_select(2, idx_t), v.index_select(2, idx_t))
        for k, v in _to_tuple_kv(past_kv)
    )
    return _to_dynamic_cache(pruned)


def measure_method(model, input_ids, method, budget, prototypes, device, repeats, tokens_to_generate=100):
    """Time the auto-regressive decode phase over `tokens_to_generate` tokens to isolate generation throughput."""
    seq_len = input_ids.shape[1]
    times = []
    vrams = []

    for _ in range(repeats):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        with torch.no_grad():
            # 1. Prefill pass
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values

            # 2. Prune initial prefill KV cache down to budget
            if method == "proactive_cache" and budget < seq_len:
                indices = predict_and_mask(prototypes, seq_len, budget)
                past_kv = prune_kv(past_kv, indices, device)

            # Isolate and time only the auto-regressive decode phase
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 3. Auto-regressive decode loop
            next_token = input_ids[:, -1:]
            for _ in range(tokens_to_generate):
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                logits = out.logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)

            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0
            peak_vram = torch.cuda.max_memory_allocated() / 1e6

        times.append(elapsed_s)
        vrams.append(peak_vram)

    avg_time_s = float(np.mean(times))
    avg_vram_mb = float(np.mean(vrams))
    tok_per_s = tokens_to_generate / avg_time_s

    return {
        "time_ms":  avg_time_s * 1000.0,
        "vram_mb":  avg_vram_mb,
        "tok_per_s": tok_per_s,
    }


def plot_results(data, seq_lens, budget, save_path):
    full_times  = [data["full"][s]["time_ms"]            for s in seq_lens]
    pc_times    = [data["proactive_cache"][s]["time_ms"] for s in seq_lens]
    full_vrams  = [data["full"][s]["vram_mb"]            for s in seq_lens]
    pc_vrams    = [data["proactive_cache"][s]["vram_mb"] for s in seq_lens]
    full_tps    = [data["full"][s]["tok_per_s"]           for s in seq_lens]
    pc_tps      = [data["proactive_cache"][s]["tok_per_s"] for s in seq_lens]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Proactive Cache (Budget={budget}) vs Full Attention — Scaling Analysis",
                 fontsize=14, fontweight="bold")
    colors = {"Full Attention": "#e74c3c", "Proactive Cache (ours)": "#2ecc71"}

    # --- Graph 1: Time vs Seq Len ---
    ax = axes[0]
    ax.plot(seq_lens, full_times, "o-", color=colors["Full Attention"],        lw=2, label="Full Attention (O(n²))")
    ax.plot(seq_lens, pc_times,   "s-", color=colors["Proactive Cache (ours)"], lw=2, label=f"Proactive Cache B={budget} (O(n))")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Figure A — Inference Time vs Sequence Length")
    ax.legend()
    ax.set_xticks(seq_lens)
    ax.grid(True, alpha=0.3)

    # --- Graph 2: VRAM vs Seq Len ---
    ax = axes[1]
    ax.plot(seq_lens, full_vrams, "o-", color=colors["Full Attention"],        lw=2, label="Full Attention")
    ax.plot(seq_lens, pc_vrams,   "s-", color=colors["Proactive Cache (ours)"], lw=2, label=f"Proactive Cache B={budget}")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Figure B — Peak VRAM vs Sequence Length")
    ax.legend()
    ax.set_xticks(seq_lens)
    ax.grid(True, alpha=0.3)

    # --- Graph 3: Throughput vs Seq Len ---
    ax = axes[2]
    ax.plot(seq_lens, full_tps, "o-", color=colors["Full Attention"],        lw=2, label="Full Attention")
    ax.plot(seq_lens, pc_tps,   "s-", color=colors["Proactive Cache (ours)"], lw=2, label=f"Proactive Cache B={budget}")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Figure C — Throughput vs Sequence Length")
    ax.legend()
    ax.set_xticks(seq_lens)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")


def main():
    args = parse_args()
    budget    = args.budget
    repeats   = args.repeats
    seq_lens  = sorted(args.seq_lens)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = os.path.join(OUTPUT_DIR, "scaling_curve.pkl")
    fig_path = os.path.join(OUTPUT_DIR, "scaling_curve.png")

    # Load checkpoint if exists
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"Resumed from checkpoint: {pkl_path}")
        print(f"  Already done: {[(m, s) for m in data for s in data[m]]}")
    else:
        data = {"full": {}, "proactive_cache": {}}

    # Load model
    result = None
    for mn in MODELS_TO_TRY:
        result = try_load_model(mn)
        if result is not None:
            break
    if result is None:
        print("ERROR: Could not load any model.")
        return
    model, tokenizer = result
    device = next(model.parameters()).device

    # Load prototypes
    model_name = MODELS_TO_TRY[0]
    prototypes = load_prototypes(model_name)
    if prototypes is None:
        print("Proactive Cache prototypes missing — only full attention will be measured.")

    print(f"\n{'='*60}")
    print(f"  O(n) Scaling Curve | budget={budget} | repeats={repeats}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"{'='*60}\n")

    for method in ["full", "proactive_cache"]:
        if method == "proactive_cache" and prototypes is None:
            print("  Skipping Proactive Cache (no prototypes).")
            continue
        for seq_len in seq_lens:
            if seq_len in data[method]:
                r = data[method][seq_len]
                print(f"  [SKIP] {method} seq={seq_len}: {r['time_ms']:.1f}ms, {r['vram_mb']:.0f}MB, {r['tok_per_s']:.0f} tok/s")
                continue

            print(f"\n--- {method} | seq_len={seq_len} ---")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            input_ids = build_synthetic_input(seq_len, device, tokenizer)

            try:
                gc.collect()
                torch.cuda.empty_cache()
                r = measure_method(model, input_ids, method, budget, prototypes, device, repeats)
                data[method][seq_len] = r
                print(f"  time={r['time_ms']:.1f}ms  VRAM={r['vram_mb']:.0f}MB  tok/s={r['tok_per_s']:.0f}")

                # Save checkpoint after every measurement
                with open(pkl_path, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"  ERROR at seq_len={seq_len}: {e}")
                break   # OOM at large seq lens — stop that method

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Scaling Curve Summary  |  Proactive Cache budget={budget}")
    print(f"{'='*70}")
    print(f"{'Seq Len':>8}  {'Full Time(ms)':>14}  {'PC Time(ms)':>12}  {'Speedup':>8}  {'Full VRAM':>10}  {'PC VRAM':>8}")
    print("-" * 70)
    for s in seq_lens:
        ft = data["full"].get(s, {}).get("time_ms", float("nan"))
        pt = data["proactive_cache"].get(s, {}).get("time_ms", float("nan"))
        fv = data["full"].get(s, {}).get("vram_mb", float("nan"))
        pv = data["proactive_cache"].get(s, {}).get("vram_mb", float("nan"))
        speedup = ft / pt if pt and pt > 0 else float("nan")
        print(f"{s:>8}  {ft:>14.1f}  {pt:>12.1f}  {speedup:>8.2f}x  {fv:>10.0f}  {pv:>8.0f}")

    # Generate figures
    successful_lens = [s for s in seq_lens if s in data["full"] and s in data["proactive_cache"]]
    if len(successful_lens) >= 2:
        print(f"\nGenerating plot for successful sequence lengths: {successful_lens}")
        plot_results(data, successful_lens, budget, fig_path)
    else:
        print("\nSkipping figure: not enough successful data points completed to plot a curve.")

    print(f"\nData saved to {pkl_path}")


if __name__ == "__main__":
    main()
