"""
benchmark_speedup.py
────────────────────
Fast wall-clock speedup measurement for all 4 models.
Measures separately:
  - TTFT   = Time-To-First-Token (prefill speedup)
  - TPOT   = Time-Per-Output-Token (decode speedup)
  - E2E    = Combined (prefill + decode) speedup
  - VRAM   = Peak GPU memory

Skips NIAH and wikitext PPL for speed.
Output: outputs/speedup/benchmark_results.json
        outputs/speedup/benchmark_table.txt
"""

import os, sys, json, time
import torch

# Force UTF-8 output so Unicode box-drawing chars in HeadGenome don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))
os.environ["HF_HOME"] = r"d:\.cache\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

from headgenome import HeadGenome
from headgenome.benchmarks.speed import measure_ttft, measure_e2e

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "speedup"))
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("openai-community/gpt2-medium", 900),   # (model_id, max_prompt_tokens)
    ("Qwen/Qwen2.5-0.5B",           2048),
    ("meta-llama/Llama-3.2-1B",     2048),
    ("Qwen/Qwen2.5-1.5B",           2048),
]

WINDOW     = 512
NEW_TOKENS = 32   # short decode for speed
WARMUP     = 2
RUNS       = 5

FILLER = (
    "The study of artificial intelligence has progressed rapidly over the past decade. "
    "Researchers have developed models capable of understanding natural language, generating "
    "images, writing code, and solving complex mathematical problems. These systems use "
    "neural networks with billions of parameters trained on vast corpora of human-written text. "
)


def build_prompt(tokenizer, max_tokens):
    ids = tokenizer(FILLER * 200, return_tensors="pt")["input_ids"][0]
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def benchmark_model(model_id, max_prompt_tokens, all_results):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu    = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n{'='*60}")
    print(f"  MODEL : {model_id}")
    print(f"  GPU   : {gpu}  |  max_prompt={max_prompt_tokens}  window={WINDOW}")
    print(f"{'='*60}")

    hg  = HeadGenome(model_id, dtype="bf16", device=device)
    tok = hg.tokenizer
    prompt = build_prompt(tok, max_prompt_tokens)
    toks   = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    print(f"  Actual prompt length: {toks} tokens")

    # ── Dense baseline ──────────────────────────────────────────
    print("\n[1/2] Dense baseline …")
    torch.cuda.reset_peak_memory_stats()
    d_ttft = measure_ttft(hg.model, tok, prompt, device=device, warmup=WARMUP, runs=RUNS)
    d_e2e  = measure_e2e (hg.model, tok, prompt, new_tokens=NEW_TOKENS,
                          device=device, warmup=WARMUP, runs=RUNS)

    print(f"  TTFT  {d_ttft['ttft_ms_mean']:.1f} ms   "
          f"prefill {d_ttft['prefill_tok_s']:,.0f} tok/s   "
          f"VRAM {d_ttft['peak_vram_gb']:.2f} GB")
    print(f"  TPOT  {d_e2e['tpot_ms']:.2f} ms/tok   "
          f"decode  {d_e2e['decode_tok_s']:.1f} tok/s   "
          f"E2E {d_e2e['e2e_latency_ms']:.0f} ms")

    # ── HeadGenome masked ───────────────────────────────────────
    print(f"\n[2/2] HeadGenome (window={WINDOW}) …")
    hg.profile(docs=30)
    hg.compile(backend="torch", window=WINDOW)
    torch.cuda.reset_peak_memory_stats()

    h_ttft = measure_ttft(hg.model, tok, prompt, device=device, warmup=WARMUP, runs=RUNS)
    h_e2e  = measure_e2e (hg.model, tok, prompt, new_tokens=NEW_TOKENS,
                          device=device, warmup=WARMUP, runs=RUNS)

    print(f"  TTFT  {h_ttft['ttft_ms_mean']:.1f} ms   "
          f"prefill {h_ttft['prefill_tok_s']:,.0f} tok/s   "
          f"VRAM {h_ttft['peak_vram_gb']:.2f} GB")
    print(f"  TPOT  {h_e2e['tpot_ms']:.2f} ms/tok   "
          f"decode  {h_e2e['decode_tok_s']:.1f} tok/s   "
          f"E2E {h_e2e['e2e_latency_ms']:.0f} ms")

    # ── Speedup ratios ──────────────────────────────────────────
    prefill_speedup = d_ttft["ttft_ms_mean"]      / max(h_ttft["ttft_ms_mean"],      1e-6)
    decode_speedup  = d_e2e["decode_latency_ms"]  / max(h_e2e["decode_latency_ms"],  1e-6)
    e2e_speedup     = d_e2e["e2e_latency_ms"]     / max(h_e2e["e2e_latency_ms"],     1e-6)
    vram_saved      = d_ttft["peak_vram_gb"]       - h_ttft["peak_vram_gb"]

    print(f"\n  ── Speedup ──")
    print(f"  Prefill (TTFT) speedup : {prefill_speedup:.2f}×")
    print(f"  Decode  (TPOT) speedup : {decode_speedup:.2f}×")
    print(f"  E2E            speedup : {e2e_speedup:.2f}×")
    print(f"  VRAM saved             : {vram_saved:+.2f} GB")

    all_results[model_id] = {
        "gpu": gpu, "prompt_tokens": toks,
        "dense":     {"ttft_ms": d_ttft["ttft_ms_mean"],
                      "prefill_tok_s": d_ttft["prefill_tok_s"],
                      "tpot_ms": d_e2e["tpot_ms"],
                      "decode_tok_s": d_e2e["decode_tok_s"],
                      "e2e_ms": d_e2e["e2e_latency_ms"],
                      "vram_gb": d_ttft["peak_vram_gb"]},
        "headgenome":{"ttft_ms": h_ttft["ttft_ms_mean"],
                      "prefill_tok_s": h_ttft["prefill_tok_s"],
                      "tpot_ms": h_e2e["tpot_ms"],
                      "decode_tok_s": h_e2e["decode_tok_s"],
                      "e2e_ms": h_e2e["e2e_latency_ms"],
                      "vram_gb": h_ttft["peak_vram_gb"]},
        "speedup":   {"prefill": round(prefill_speedup, 3),
                      "decode":  round(decode_speedup,  3),
                      "e2e":     round(e2e_speedup,     3),
                      "vram_saved_gb": round(vram_saved, 3)},
    }

    hg.remove()
    del hg
    torch.cuda.empty_cache()
    return all_results[model_id]


def print_summary(all_results):
    W = 90
    print("\n" + "═"*W)
    print(f"  {'Model':<30} {'TTFT Dense':>10} {'TTFT HG':>9} {'Prefill↑':>9}"
          f" {'TPOT Dense':>11} {'TPOT HG':>9} {'Decode↑':>8} {'E2E↑':>7} {'VRAM':>6}")
    print("─"*W)
    for mid, r in all_results.items():
        d, h, sp = r["dense"], r["headgenome"], r["speedup"]
        name = mid.split("/")[-1]
        print(f"  {name:<30} {d['ttft_ms']:>9.1f}ms {h['ttft_ms']:>8.1f}ms {sp['prefill']:>8.2f}×"
              f" {d['tpot_ms']:>10.2f}ms {h['tpot_ms']:>8.2f}ms {sp['decode']:>7.2f}×"
              f" {sp['e2e']:>6.2f}× {sp['vram_saved_gb']:>+5.2f}G")
    print("═"*W)


def main():
    all_results = {}
    for model_id, max_tok in MODELS:
        try:
            benchmark_model(model_id, max_tok, all_results)
        except torch.cuda.OutOfMemoryError:
            print(f"[SKIP] {model_id} — OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERROR] {model_id}: {e}")
            import traceback; traceback.print_exc()
            try: torch.cuda.empty_cache()
            except: pass

    if all_results:
        print_summary(all_results)

    out = os.path.join(OUT_DIR, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Saved → {out}")

    # Also write a plain-text summary table
    txt = os.path.join(OUT_DIR, "benchmark_table.txt")
    with open(txt, "w") as f:
        f.write("Model,TTFT_Dense_ms,TTFT_HG_ms,Prefill_Speedup,TPOT_Dense_ms,TPOT_HG_ms,Decode_Speedup,E2E_Speedup,VRAM_Saved_GB\n")
        for mid, r in all_results.items():
            d, h, sp = r["dense"], r["headgenome"], r["speedup"]
            f.write(f"{mid},{d['ttft_ms']:.2f},{h['ttft_ms']:.2f},{sp['prefill']:.3f},"
                    f"{d['tpot_ms']:.3f},{h['tpot_ms']:.3f},{sp['decode']:.3f},{sp['e2e']:.3f},{sp['vram_saved_gb']:.3f}\n")
    print(f"[Done] Table → {txt}")


if __name__ == "__main__":
    main()
