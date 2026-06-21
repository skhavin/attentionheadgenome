# KVPress Standard Benchmark Runner — Compares Proactive Cache against KVPress built-in presses
# using the KVPress standard evaluation protocol on LLaMA 3.1 8B.
#
# Benchmarks:
#   - ProactiveCachePress (ours)
#   - StreamingLLMPress (KVPress built-in)
#   - SnapKVPress (KVPress built-in)
#   - KnormPress (KVPress built-in)
#   - Full Attention baseline
#
# Evaluation suites:
#   - Perplexity on WikiText-103 long docs (local, fast)
#   - LongBench (requires kvpress[longbench] — optional)
#
# Usage:
#   pip install kvpress                    # core
#   pip install kvpress[longbench]         # optional: for LongBench suite
#   python phase6/kvpress_benchmark.py
#   python phase6/kvpress_benchmark.py --compression-ratio 0.75 --suite ppl
#
# Output:
#   outputs/phase6/kvpress_results.pkl
#   outputs/phase6/kvpress_results.md   (markdown table, paste into paper)

# Shim for Python 3.13+ which removed the 'pipes' module (needed by fire/kvpress)
try:
    import pipes
except ImportError:
    import sys, shlex
    sys.modules['pipes'] = shlex

import sys, os, argparse
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data_utils import load_concatenated_articles
from config import PHASE4_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase6")
PHASE4_DIR_ABS = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase4")

MODELS_TO_TRY = ["unsloth/meta-llama-3.1-8B-bnb-4bit"]
NUM_DOCS      = 2
SEQ_LEN       = 128
EVAL_TOKENS   = 15

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def parse_args():
    parser = argparse.ArgumentParser(description="KVPress standard benchmark")
    parser.add_argument("--compression-ratio", type=float, default=0.75,
                        help="Fraction of KV cache to evict (default: 0.75 = keep 25%)")
    parser.add_argument("--suite", choices=["ppl", "longbench", "both"], default="ppl",
                        help="Evaluation suite to run")
    return parser.parse_args()


def get_proto_path(model_name):
    short = model_name.split("/")[-1].lower()
    return os.path.join(PHASE4_DIR_ABS, f"{short}_prototypes.pkl")


OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")

def try_load_model(model_name):
    """Tries GPU-only first, then auto, then disk offload."""
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


def eval_ppl_with_kv(model, input_ids, past_kv, eval_start):
    targets  = input_ids[:, eval_start:]
    gen_len  = targets.shape[1]
    if gen_len < 5:
        return None
    nlls = []
    next_tok = input_ids[:, eval_start - 1:eval_start]
    for i in range(gen_len):
        out      = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        logits   = out.logits[:, -1, :]
        target   = targets[:, i]
        nll      = torch.nn.functional.cross_entropy(logits, target).item()
        nlls.append(nll)
        next_tok = target.unsqueeze(0)
    return float(np.exp(np.mean(nlls)))


# ------------------------------------------------------------------
# KVPress-based evaluation loop
# ------------------------------------------------------------------

def eval_ppl_with_press(model, tokenizer, docs, press, device):
    """
    Standard PPL evaluation using a KVPress press object.
    Uses the press as a context manager to hook into the forward pass.
    """
    import gc
    ppls = []
    for text in docs:
        gc.collect()
        torch.cuda.empty_cache()
        
        tokens    = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=SEQ_LEN)
        input_ids = tokens["input_ids"].to(device)
        seq_len   = input_ids.shape[1]
        eval_start = seq_len - min(EVAL_TOKENS, seq_len // 4)
        if eval_start < 20:
            continue
        context = input_ids[:, :eval_start - 1]

        try:
            with torch.no_grad():
                if press is None:
                    # Full attention baseline
                    out = model(context, use_cache=True)
                    past_kv = out.past_key_values
                else:
                    # KVPress context manager hooks into the forward pass
                    with press(model):
                        out = model(context, use_cache=True)
                        past_kv = out.past_key_values

            ppl = eval_ppl_with_kv(model, input_ids, past_kv, eval_start)
            if ppl:
                ppls.append(ppl)
        except Exception as e:
            print(f"    [ERROR inside document iteration]: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    gc.collect()
    torch.cuda.empty_cache()
    return float(np.mean(ppls)) if ppls else float("nan")


def build_press_roster(compression_ratio, proto_path):
    """
    Build the list of (label, press) pairs to benchmark.
    Gracefully skips any press that isn't installed.
    """
    from phase6.proactive_cache_press import ProactiveCachePress
    roster = [("Full Attention", None)]

    roster.append(("Proactive Cache (ours)", ProactiveCachePress(
        compression_ratio=compression_ratio,
        prototype_path=proto_path,
    )))

    try:
        from kvpress import StreamingLLMPress
        roster.append(("StreamingLLM", StreamingLLMPress(compression_ratio=compression_ratio)))
    except ImportError:
        print("  [SKIP] StreamingLLMPress not available (install kvpress)")

    try:
        from kvpress import SnapKVPress
        roster.append(("SnapKV", SnapKVPress(compression_ratio=compression_ratio)))
    except ImportError:
        print("  [SKIP] SnapKVPress not available")

    try:
        from kvpress import KnormPress
        roster.append(("KNorm", KnormPress(compression_ratio=compression_ratio)))
    except ImportError:
        print("  [SKIP] KnormPress not available")

    try:
        from kvpress import ExpectedAttentionPress
        roster.append(("ExpectedAttn", ExpectedAttentionPress(compression_ratio=compression_ratio)))
    except ImportError:
        pass

    return roster


def write_markdown_table(results, save_path, compression_ratio):
    """Write results as a markdown table suitable for copy-paste into the paper."""
    lines = [
        f"## KVPress Standard Benchmark (LLaMA-3.1-8B, compression_ratio={compression_ratio})\n",
        "| Method | PPL ↓ | VRAM (MB) | Time (s) |",
        "|---|---|---|---|",
    ]
    for label, r in results.items():
        lines.append(f"| {label} | {r['ppl']:.2f} | {r['vram_mb']:.0f} | {r['time_s']:.1f} |")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown table saved to {save_path}")


def run_ppl_suite(model, tokenizer, docs, roster, device, pkl_path, compression_ratio):
    """Run the PPL suite across all presses and save results."""
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
        print(f"  Loaded existing results from {pkl_path}")
    else:
        results = {}

    for label, press in roster:
        if label in results:
            r = results[label]
            print(f"  [SKIP] {label}: PPL={r['ppl']:.2f} (already done)")
            continue

        print(f"\n--- {label} | compression_ratio={compression_ratio} ---")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ppl = eval_ppl_with_press(model, tokenizer, docs, press, device)
        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e6

        results[label] = {"ppl": ppl, "vram_mb": vram, "time_s": elapsed}
        print(f"  PPL={ppl:.2f}  VRAM={vram:.0f}MB  Time={elapsed:.1f}s")

        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)

    return results


def main():
    args             = parse_args()
    compression_ratio = args.compression_ratio
    suite            = args.suite

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = os.path.join(OUTPUT_DIR, "kvpress_results.pkl")
    md_path  = os.path.join(OUTPUT_DIR, "kvpress_results.md")

    # Check kvpress is installed
    try:
        import kvpress
        print("kvpress library detected.")
    except Exception as e:
        print("ERROR: kvpress is not installed or failed to import.")
        print(f"  Error details: {e}")
        print("  Install with: pip install kvpress")
        print("  Then re-run this script.")
        return

    # Load model
    result = None
    for mn in MODELS_TO_TRY:
        result = try_load_model(mn)
        if result is not None:
            model_name = mn
            break
    if result is None:
        print("ERROR: Could not load any model.")
        return
    model, tokenizer = result
    device = next(model.parameters()).device

    proto_path = get_proto_path(model_name)
    roster     = build_press_roster(compression_ratio, proto_path)

    print(f"\n{'='*60}")
    print(f"  KVPress Benchmark | compression_ratio={compression_ratio}")
    print(f"  Suite: {suite} | Presses: {[l for l, _ in roster]}")
    print(f"{'='*60}\n")

    if suite in ("ppl", "both"):
        docs = load_concatenated_articles(split="validation",
                                          articles_per_doc=10, max_docs=NUM_DOCS)
        results = run_ppl_suite(model, tokenizer, docs, roster, device, pkl_path, compression_ratio)

        # Print final table
        print(f"\n{'='*55}")
        print(f"  KVPress PPL Results | compression_ratio={compression_ratio}")
        print(f"{'='*55}")
        print(f"{'Method':<20} {'PPL':>8} {'VRAM(MB)':>10} {'Time(s)':>9}")
        print("-" * 55)
        for label, r in results.items():
            print(f"{label:<20} {r['ppl']:>8.2f} {r['vram_mb']:>10.0f} {r['time_s']:>9.1f}")

        write_markdown_table(results, md_path, compression_ratio)

    if suite in ("longbench", "both"):
        try:
            from kvpress.eval.longbench import run_longbench
            print("\n--- Running LongBench (this may take 30-60 min) ---")
            run_longbench(model, tokenizer, roster, OUTPUT_DIR)
        except ImportError:
            print("\n[SKIP] LongBench not available.")
            print("  Install with: pip install kvpress[longbench]")


if __name__ == "__main__":
    main()
