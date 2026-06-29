"""
measure_speedup.py
──────────────────
Measures and compares Dense vs HeadGenome (Torch Mask) TTFT, TPOT,
E2E latency, VRAM, PPL, and NIAH for all locally cached models that fit in VRAM.

Produces:
  outputs/speedup/speedup_results.json
  outputs/speedup/speedup_table.txt

Usage:
  python measure_speedup.py
"""

import os, sys, json, time
import torch

# Allow importing from lib/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))

os.environ["HF_HOME"]          = r"d:\.cache\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

from headgenome import HeadGenome
from headgenome.benchmarks.speed import measure_ttft, measure_e2e
from headgenome.benchmarks.ppl   import measure_ppl
from headgenome.benchmarks.niah  import run_niah

OUT_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "outputs", "speedup")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Models to test (in order of size) ─────────────────────────────────────────
MODELS = [
    "openai-community/gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
]

SEQ_LEN    = 4096
NEW_TOKENS = 64          # Keep short for speed on 4GB VRAM
WINDOW     = 512
WARMUP     = 3
RUNS       = 10
NIAH_N     = 5           # Samples per depth for quick run


def build_prompt(tokenizer, target_len: int) -> str:
    FILLER = (
        "The study of artificial intelligence has progressed rapidly over the past decade. "
        "Researchers have developed models capable of understanding natural language, generating "
        "images, writing code, and solving complex mathematical problems. These systems use "
        "neural networks with billions of parameters trained on vast corpora of human-written text. "
    )
    ids = tokenizer(FILLER * 500, return_tensors="pt")["input_ids"][0]
    truncated = ids[:target_len]
    return tokenizer.decode(truncated, skip_special_tokens=True)


def run_model(model_id: str, all_results: dict):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*64}")
    print(f"  MODEL: {model_id}")
    print(f"  GPU:   {gpu_name}")
    print(f"  SEQ:   {SEQ_LEN} tokens | WINDOW: {WINDOW}")
    print(f"{'='*64}")

    hg = HeadGenome(model_id, dtype="bf16", device=device)
    tok = hg.tokenizer

    target_seq = SEQ_LEN
    if "gpt2" in model_id.lower():
        target_seq = 950

    prompt = build_prompt(tok, target_seq)
    actual_len = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    print(f"  Prompt length: {actual_len} tokens")

    model_results = {"model_id": model_id, "gpu": gpu_name, "seq_len": actual_len}

    # ── 1. Dense Baseline ──────────────────────────────────────────────────
    print(f"\n[1/3] Dense Baseline")
    torch.cuda.reset_peak_memory_stats()

    haystack = 300
    if "gpt2" in model_id.lower():
        haystack = 50

    dense_ttft = measure_ttft(hg.model, tok, prompt, device=device, warmup=WARMUP, runs=RUNS)
    dense_e2e  = measure_e2e(hg.model, tok, prompt, new_tokens=NEW_TOKENS, device=device, warmup=WARMUP, runs=max(RUNS//2, 3))
    dense_ppl  = measure_ppl(hg.model, tok, seq_len=512, device=device)
    dense_niah = run_niah(hg.model, tok, num_samples=NIAH_N, haystack_sentences=haystack, device=device)

    model_results["dense"] = {
        "ttft_ms_mean":   dense_ttft["ttft_ms_mean"],
        "ttft_ms_median": dense_ttft["ttft_ms_median"],
        "ttft_ms_p90":    dense_ttft["ttft_ms_p90"],
        "prefill_tok_s":  dense_ttft["prefill_tok_s"],
        "tpot_ms":        dense_e2e["tpot_ms"],
        "decode_tok_s":   dense_e2e["decode_tok_s"],
        "e2e_latency_ms": dense_e2e["e2e_latency_ms"],
        "peak_vram_gb":   dense_ttft["peak_vram_gb"],
        "ppl":            dense_ppl["ppl"],
        "niah_pct":       dense_niah["overall_pct"],
    }

    print(f"   TTFT (mean/median/p90): {dense_ttft['ttft_ms_mean']:.1f} / {dense_ttft['ttft_ms_median']:.1f} / {dense_ttft['ttft_ms_p90']:.1f} ms")
    print(f"   Prefill tok/s:          {dense_ttft['prefill_tok_s']:,.0f}")
    print(f"   TPOT:                   {dense_e2e['tpot_ms']:.1f} ms")
    print(f"   E2E:                    {dense_e2e['e2e_latency_ms']:.0f} ms")
    print(f"   Peak VRAM:              {dense_ttft['peak_vram_gb']:.2f} GB")
    print(f"   PPL:                    {dense_ppl['ppl']}")
    print(f"   NIAH:                   {dense_niah['overall_pct']:.1f}%")

    # ── 2. Profile + Patch ─────────────────────────────────────────────────
    print(f"\n[2/3] HeadGenome Torch Mask (W={WINDOW})")
    hg.profile(docs=50)
    hg.compile(backend="torch", window=WINDOW)

    # ── 3. HeadGenome Benchmark ────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()

    hg_ttft = measure_ttft(hg.model, tok, prompt, device=device, warmup=WARMUP, runs=RUNS)
    hg_e2e  = measure_e2e(hg.model, tok, prompt, new_tokens=NEW_TOKENS, device=device, warmup=WARMUP, runs=max(RUNS//2, 3))
    hg_ppl  = measure_ppl(hg.model, tok, seq_len=512, device=device)
    hg_niah = run_niah(hg.model, tok, num_samples=NIAH_N, haystack_sentences=haystack, device=device)

    model_results["headgenome"] = {
        "ttft_ms_mean":   hg_ttft["ttft_ms_mean"],
        "ttft_ms_median": hg_ttft["ttft_ms_median"],
        "ttft_ms_p90":    hg_ttft["ttft_ms_p90"],
        "prefill_tok_s":  hg_ttft["prefill_tok_s"],
        "tpot_ms":        hg_e2e["tpot_ms"],
        "decode_tok_s":   hg_e2e["decode_tok_s"],
        "e2e_latency_ms": hg_e2e["e2e_latency_ms"],
        "peak_vram_gb":   hg_ttft["peak_vram_gb"],
        "ppl":            hg_ppl["ppl"],
        "niah_pct":       hg_niah["overall_pct"],
    }

    ttft_speedup = dense_ttft["ttft_ms_mean"] / max(hg_ttft["ttft_ms_mean"], 1)
    e2e_speedup  = dense_e2e["e2e_latency_ms"] / max(hg_e2e["e2e_latency_ms"], 1)
    vram_delta   = dense_ttft["peak_vram_gb"] - hg_ttft["peak_vram_gb"]

    model_results["speedup"] = {
        "ttft_speedup": round(ttft_speedup, 3),
        "e2e_speedup":  round(e2e_speedup, 3),
        "vram_saved_gb": round(vram_delta, 3),
    }

    print(f"   TTFT (mean/median/p90): {hg_ttft['ttft_ms_mean']:.1f} / {hg_ttft['ttft_ms_median']:.1f} / {hg_ttft['ttft_ms_p90']:.1f} ms")
    print(f"   Prefill tok/s:          {hg_ttft['prefill_tok_s']:,.0f}")
    print(f"   TPOT:                   {hg_e2e['tpot_ms']:.1f} ms")
    print(f"   E2E:                    {hg_e2e['e2e_latency_ms']:.0f} ms")
    print(f"   Peak VRAM:              {hg_ttft['peak_vram_gb']:.2f} GB")
    print(f"   PPL:                    {hg_ppl['ppl']}")
    print(f"   NIAH:                   {hg_niah['overall_pct']:.1f}%")
    print(f"   ── Speedup ──────────────────────────────────")
    print(f"   TTFT speedup:           {ttft_speedup:.2f}×  (measured wall-clock)")
    print(f"   E2E speedup:            {e2e_speedup:.2f}×")
    print(f"   VRAM saved:             {vram_delta:+.2f} GB")

    all_results[model_id] = model_results

    # Print comparison table
    _print_table(model_id, model_results)

    hg.remove()
    del hg.model
    torch.cuda.empty_cache()


def _print_table(model_id: str, r: dict):
    d  = r["dense"]
    hg = r["headgenome"]
    sp = r["speedup"]

    w = 64
    print()
    print("═" * w)
    print(f"  Benchmark Table: {model_id}")
    print(f"  Seq len: {r['seq_len']}  |  GPU: {r['gpu']}")
    print("─" * w)
    header = f"{'Method':<22}{'TTFT(ms)':>10}{'Prefill t/s':>13}{'TPOT(ms)':>10}{'E2E(ms)':>10}{'VRAM(GB)':>10}{'NIAH%':>8}{'PPL':>8}"
    print(header)
    print("─" * w)
    row_dns = f"{'Dense SDPA':<22}{d['ttft_ms_mean']:>10.1f}{d['prefill_tok_s']:>13,.0f}{d['tpot_ms']:>10.1f}{d['e2e_latency_ms']:>10.0f}{d['peak_vram_gb']:>10.2f}{d['niah_pct']:>8.1f}{d['ppl']:>8.2f}"
    row_hg  = f"{'HG Torch Mask':<22}{hg['ttft_ms_mean']:>10.1f}{hg['prefill_tok_s']:>13,.0f}{hg['tpot_ms']:>10.1f}{hg['e2e_latency_ms']:>10.0f}{hg['peak_vram_gb']:>10.2f}{hg['niah_pct']:>8.1f}{hg['ppl']:>8.2f}"
    print(row_dns)
    print(row_hg)
    print("─" * w)
    print(f"  TTFT Speedup: {sp['ttft_speedup']:.2f}×   E2E Speedup: {sp['e2e_speedup']:.2f}×   VRAM saved: {sp['vram_saved_gb']:+.2f} GB")
    print("═" * w)


def main():
    all_results = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
        "settings": {"seq_len": SEQ_LEN, "window": WINDOW, "new_tokens": NEW_TOKENS, "runs": RUNS},
        "models": {}
    }

    for model_id in MODELS:
        try:
            run_model(model_id, all_results["models"])
        except torch.cuda.OutOfMemoryError:
            print(f"\n[SKIP] {model_id} — CUDA OOM. Skipping.")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[ERROR] {model_id}: {e}")
            torch.cuda.empty_cache()

    out_json = os.path.join(OUT_DIR, "speedup_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Results saved -> {out_json}")


if __name__ == "__main__":
    main()
