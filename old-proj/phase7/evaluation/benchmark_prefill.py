# phase7/benchmark_prefill.py
#
# Phase 6 — Latency / Compute Benchmarks (Prefill Speedup)
#
# Measures wall-clock prefill time for full attention vs. the mixed-complexity
# substitution: O(1) for sink heads, O(N·W) for local heads, O(N²) only for
# content/global heads.
#
# This is the key novelty over DuoAttention/MoA/FastGen: the substitution
# changes the compute graph, not just the KV cache. The speedup should be
# visible in the prefill phase even without KV eviction.
#
# What we measure:
#   full_attention  — exact softmax, O(N²) for all heads
#   substitute_tier1— Tier 1 heads use O(1)/O(N·W) substitution
#   substitute_all  — Tier 1 + Tier 2 heads substituted (upper-bound speedup)
#
# Output: outputs/phase7/prefill_latency.json + a latency-vs-seq_len plot
#
# Usage:
#   python phase7/benchmark_prefill.py
#   python phase7/benchmark_prefill.py --seq_lens 512 1024 2048 4096 --n_runs 20

import sys, os, argparse, json, time, pickle
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from config import PHASE7_DIR

OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 — Prefill latency benchmarks")
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--seq_lens", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096],
                   help="Sequence lengths to benchmark (default: 512 1024 2048 4096)")
    p.add_argument("--n_runs", type=int, default=10,
                   help="Number of timed runs per (mode, seq_len) (default: 10)")
    p.add_argument("--n_warmup", type=int, default=3,
                   help="Warmup runs before timing (default: 3)")
    p.add_argument("--audit_path",
                   default=os.path.join(PHASE7_DIR, "head_audit.pkl"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--plot", action="store_true",
                   help="Generate latency-vs-seq_len plot (requires matplotlib)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    strategies = [
        ({"device_map": {"": device}}, "single-device"),
        ({"device_map": "auto"},       "auto"),
        ({"device_map": "auto", "offload_folder": OFFLOAD_DIR,
          "offload_state_dict": True}, "disk-offload"),
    ]
    for extra, tag in strategies:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                attn_implementation="eager", **extra)
            model.eval()
            print(f"  Loaded [{tag}]")
            return model, tokenizer
        except Exception as e:
            print(f"  Failed [{tag}]: {str(e)[:100]}")
    raise RuntimeError(f"Could not load {model_name}")


# ---------------------------------------------------------------------------
# Tier lists
# ---------------------------------------------------------------------------

def load_tier_lists(audit_path):
    if not os.path.exists(audit_path):
        print(f"  WARNING: {audit_path} not found. Run audit_heads.py first.")
        return [], []
    with open(audit_path, "rb") as f:
        audit = pickle.load(f)
    tier1 = [(r["layer"], r["head"], r["type"]) for r in audit["rows"] if r.get("tier") == 1]
    tier2 = [(r["layer"], r["head"], r["type"]) for r in audit["rows"] if r.get("tier") == 2]
    return tier1, tier2


# ---------------------------------------------------------------------------
# Prefill timing
# ---------------------------------------------------------------------------

def time_prefill(model, tokens, device, n_runs, n_warmup):
    """
    Measure median wall-clock prefill time over n_runs.

    Warmup runs are excluded from timing to avoid CUDA JIT overhead.
    Uses torch.cuda.synchronize() to ensure GPU ops are complete before timing.
    """
    model_device = next(model.parameters()).device
    tokens = tokens.to(model_device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(tokens, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(tokens, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_s":   float(np.mean(times)),
        "median_s": float(np.median(times)),
        "std_s":    float(np.std(times)),
        "min_s":    float(np.min(times)),
        "max_s":    float(np.max(times)),
    }


# ---------------------------------------------------------------------------
# Complexity annotation
# ---------------------------------------------------------------------------

def describe_complexity(mode, num_heads, tier1_count, tier2_count):
    """
    Return a human-readable complexity string for the substitution mode.
    This is the key contribution: mixed-complexity prefill.
    """
    if mode == "full_attention":
        return f"O(N²) × {num_heads} heads"
    elif mode == "substitute_tier1":
        full_heads = num_heads - tier1_count
        sub1 = tier1_count
        return f"O(N²) × {full_heads} + O(1)/O(N·W) × {sub1}"
    else:  # substitute_all
        full_heads = num_heads - tier1_count - tier2_count
        sub12 = tier1_count + tier2_count
        return f"O(N²) × {full_heads} + O(1)/O(N·W) × {sub12}"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_latency_plot(latency_data, seq_lens, output_dir):
    """
    Generate a latency-vs-seq_len plot showing the three complexity curves.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "full_attention":      "#e74c3c",
        "substitute_tier1":    "#3498db",
        "substitute_all":      "#2ecc71",
    }
    labels = {
        "full_attention":   "Full Attention O(N²)",
        "substitute_tier1": "Tier 1 Substituted",
        "substitute_all":   "Tier 1+2 Substituted",
    }

    for mode, color in colors.items():
        x_vals, y_vals = [], []
        for N in seq_lens:
            key = f"{mode}_sl{N}"
            if key in latency_data:
                x_vals.append(N)
                y_vals.append(latency_data[key]["median_s"] * 1000)  # ms
        if x_vals:
            ax1.plot(x_vals, y_vals, "o-", color=color, label=labels[mode], linewidth=2)

    ax1.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax1.set_ylabel("Prefill Latency (ms)", fontsize=12)
    ax1.set_title("Prefill Latency vs. Sequence Length", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Speedup plot
    for mode in ["substitute_tier1", "substitute_all"]:
        x_vals, y_vals = [], []
        for N in seq_lens:
            full_key = f"full_attention_sl{N}"
            sub_key  = f"{mode}_sl{N}"
            if full_key in latency_data and sub_key in latency_data:
                full_t = latency_data[full_key]["median_s"]
                sub_t  = latency_data[sub_key]["median_s"]
                x_vals.append(N)
                y_vals.append(full_t / sub_t)
        if x_vals:
            ax2.plot(x_vals, y_vals, "s-", color=colors[mode],
                     label=labels[mode], linewidth=2)

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1× (no speedup)")
    ax2.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax2.set_ylabel("Speedup vs. Full Attention", fontsize=12)
    ax2.set_title("Speedup: Substitution vs. Full Attention", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "prefill_latency.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(PHASE7_DIR, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  Phase 6 — Prefill Latency | model={args.model}")
    print(f"  seq_lens={args.seq_lens} | n_runs={args.n_runs} | n_warmup={args.n_warmup}")
    print(f"  device={device}")
    print(f"{'='*70}\n")

    model, tokenizer = load_model(args.model, str(device))
    device_actual = next(model.parameters()).device

    tier1, tier2 = load_tier_lists(args.audit_path)
    num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    num_heads  = model.config.n_head  if hasattr(model.config, "n_head")  else model.config.num_attention_heads
    total_heads = num_layers * num_heads

    print(f"  Model: {num_layers} layers × {num_heads} heads = {total_heads} total heads")
    print(f"  Tier 1 (safe substitution): {len(tier1)}")
    print(f"  Tier 2 (regime-switching):  {len(tier2)}")
    print(f"  Tier 3 (full attention):    {total_heads - len(tier1) - len(tier2)}")

    latency_data = {}
    save_path = os.path.join(PHASE7_DIR, "prefill_latency.json")

    modes = ["full_attention", "substitute_tier1", "substitute_all"]

    for N in args.seq_lens:
        # Random token ids — benchmarking compute graph, not specific content
        tokens = torch.randint(0, 50257, (1, N)).to(device_actual)

        for mode in modes:
            key = f"{mode}_sl{N}"
            print(f"\n  Timing: {mode} | seq_len={N}")

            # Apply patcher
            restore_fn = None
            if mode == "substitute_tier1" and tier1:
                from phase7.regime_detector import RegimeSwitchingPatcher
                patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1, tier2_heads=[])
                restore_fn = patcher.restore
            elif mode == "substitute_all" and (tier1 or tier2):
                from phase7.regime_detector import RegimeSwitchingPatcher
                patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1,
                                                  tier2_heads=tier2)
                restore_fn = patcher.restore

            timing = time_prefill(model, tokens, device_actual,
                                  n_runs=args.n_runs, n_warmup=args.n_warmup)

            if restore_fn:
                restore_fn()

            full_key = f"full_attention_sl{N}"
            speedup  = (latency_data[full_key]["median_s"] / timing["median_s"]
                        if full_key in latency_data else None)

            complexity = describe_complexity(
                mode, total_heads, len(tier1), len(tier2))

            latency_data[key] = {
                **timing,
                "mode":       mode,
                "seq_len":    N,
                "speedup":    speedup,
                "complexity": complexity,
            }

            print(f"    median={timing['median_s']*1000:.1f}ms  "
                  f"mean={timing['mean_s']*1000:.1f}ms  "
                  f"std={timing['std_s']*1000:.1f}ms  "
                  + (f"speedup={speedup:.2f}×" if speedup else ""))

    # ---- Save ----
    with open(save_path, "w") as f:
        json.dump(latency_data, f, indent=2)
    print(f"\n  Latency data saved to: {save_path}")

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print(f"  PREFILL LATENCY SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Mode':<30} {'SeqLen':>7} {'Median(ms)':>11} {'Speedup':>9} {'Complexity'}")
    print(f"  {'-'*78}")
    for N in args.seq_lens:
        for mode in modes:
            key = f"{mode}_sl{N}"
            if key in latency_data:
                r = latency_data[key]
                sp = f"{r['speedup']:.2f}×" if r.get("speedup") else "baseline"
                print(f"  {mode:<30} {N:>7} {r['median_s']*1000:>11.1f} "
                      f"{sp:>9}  {r['complexity']}")
        print(f"  {'-'*78}")

    print(f"\n  KEY CLAIM: O(1) sink + O(N·W) local heads reduce the PREFILL compute")
    print(f"  for the substituted fraction, producing mixed-complexity behaviour.")
    print(f"  This is distinct from DuoAttention/MoA/FastGen which reduce KV cache")
    print(f"  size but not the O(N²) prefill compute for every head.")

    # ---- Optional plot ----
    if args.plot:
        make_latency_plot(latency_data, args.seq_lens, PHASE7_DIR)


if __name__ == "__main__":
    main()
