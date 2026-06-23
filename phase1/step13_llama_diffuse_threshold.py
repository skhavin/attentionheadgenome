# -*- coding: utf-8 -*-
# phase1/step13_llama_diffuse_threshold.py
#
# PURPOSE: Address Gap 4 - Does Llama-3.2-1B truly lack retrieval heads,
#          or does it distribute retrieval function diffusely across many heads?
#
# METHOD:
#   Re-use the saved llama1b_retrieval_entropy.json (20 pairs already run).
#   Sweep thresholds from 0.10 to 0.35 and count emerging retrieval/induction heads.
#   Hypothesis test:
#     "Absent" = head count stays near 0 even at delta > 0.15 → genuine null
#     "Diffuse" = many heads emerge at 0.15-0.20 → widely distributed
#
# OUTPUTS:
#   outputs/phase1/llama_diffuse_threshold.json

import os
import json
import numpy as np

os.environ["HF_HOME"]          = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")

# Also try loading the 50-pair robust result if it exists
LLAMA_FILES = [
    os.path.join(OUT_DIR, "robust_entropy_llama1b.json"),    # 50-pair (preferred)
    os.path.join(OUT_DIR, "llama1b_retrieval_entropy.json"),  # 20-pair fallback
]

THRESHOLDS_RET = [0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.27, 0.30, 0.35]
THRESHOLDS_IND = [-0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50, -0.55, -0.60]


def load_data():
    for fpath in LLAMA_FILES:
        if os.path.exists(fpath):
            print(f"Loading: {fpath}")
            with open(fpath) as f:
                return json.load(f), fpath
    raise FileNotFoundError(
        "No Llama entropy data found. Run step11_retrieval_entropy_llama.py first."
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data, source = load_data()
    heads = data["heads"]
    arch  = data.get("architecture", {})
    n_pairs = data.get("n_pairs", "?")

    # Extract clean deltas (exclude NaN/sink)
    deltas = {}
    for key, v in heads.items():
        if v.get("nan") or v.get("delta") is None:
            continue
        deltas[key] = v["delta"]

    arr = np.array(list(deltas.values()))
    total_heads = len(heads)
    clean_heads = len(deltas)

    print(f"\nLlama-3.2-1B Diffuse Retrieval Analysis")
    print(f"  Source file:  {os.path.basename(source)}")
    print(f"  Prompt pairs: {n_pairs}")
    print(f"  Architecture: {arch}")
    print(f"  Total heads:  {total_heads}  |  Non-NaN heads: {clean_heads}")
    print(f"\n  Delta distribution:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
    print(f"    p50={np.percentile(arr,50):.4f}  p75={np.percentile(arr,75):.4f}")
    print(f"    p90={np.percentile(arr,90):.4f}  p95={np.percentile(arr,95):.4f}")
    print(f"    max={arr.max():.4f}  min={arr.min():.4f}")

    # Retrieval sweep
    print(f"\n  === Retrieval Threshold Sweep ===")
    print(f"  {'Threshold':>12}  {'N heads':>8}  {'% of clean':>10}  Verdict")
    print(f"  {'-'*50}")
    ret_results = {}
    prev = None
    for thr in THRESHOLDS_RET:
        n = int((arr > thr).sum())
        pct = 100 * n / clean_heads
        # Detect "elbow" where counts drop sharply
        jump = "" if prev is None else f"  (drop: {prev-n})" if prev - n > 5 else ""
        verdict = ""
        if n == 0:
            verdict = "<-- ABSENT"
        elif n <= 3:
            verdict = "<-- NEAR ABSENT"
        elif n <= 15:
            verdict = "<-- DIFFUSE"
        else:
            verdict = "<-- WIDESPREAD"
        marker = " <-- original baseline" if abs(thr - 0.30) < 0.001 else ""
        print(f"  {thr:>12.2f}  {n:>8}  {pct:>9.1f}%  {verdict}{jump}{marker}")
        ret_results[str(thr)] = {"n_retrieval": n, "pct": round(pct, 2)}
        prev = n

    # Induction sweep
    print(f"\n  === Induction Threshold Sweep ===")
    print(f"  {'Threshold':>12}  {'N heads':>8}  {'% of clean':>10}")
    print(f"  {'-'*40}")
    ind_results = {}
    for thr in THRESHOLDS_IND:
        n = int((arr < thr).sum())
        pct = 100 * n / clean_heads
        marker = " <-- original baseline" if abs(thr + 0.50) < 0.001 else ""
        print(f"  {thr:>12.2f}  {n:>8}  {pct:>9.1f}%{marker}")
        ind_results[str(thr)] = {"n_induction": n, "pct": round(pct, 2)}

    # Scientific conclusion
    count_at_015 = int((arr > 0.15).sum())
    count_at_020 = int((arr > 0.20).sum())
    count_at_030 = int((arr > 0.30).sum())

    print(f"\n  === Scientific Conclusion ===")
    if count_at_015 <= 5:
        conclusion = "GENUINE NULL: Llama-3.2-1B lacks dedicated retrieval heads at all tested thresholds. Retrieval is either fully distributed or handled by induction mechanics."
        verdict = "genuine_null"
    elif count_at_020 > 10:
        conclusion = f"DIFFUSE RETRIEVAL: {count_at_020} heads at delta>0.20 vs {count_at_030} at delta>0.30. Llama distributes retrieval broadly — GQA group sharing prevents single-head specialization."
        verdict = "diffuse"
    else:
        conclusion = f"MARGINAL: {count_at_020} heads at delta>0.20. Borderline case — GQA may partially suppress retrieval specialization."
        verdict = "marginal"
    print(f"  {conclusion}")

    # Top 20 heads by delta regardless of threshold (raw ranking)
    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  === Top 20 Heads by Delta (no threshold) ===")
    print(f"  {'rank':<6}  {'head':>8}  {'delta':>8}  {'match_ent':>10}  {'nonmatch_ent':>12}  kv_grp")
    for rank, (key, delta) in enumerate(sorted_deltas[:20]):
        v = heads[key]
        kv = v.get("kv_group", "?")
        me = v.get("match_entropy", float("nan"))
        nme = v.get("nonmatch_entropy", float("nan"))
        print(f"  {rank+1:<6}  {key:>8}  {delta:>8.4f}  {me:>10.4f}  {nme:>12.4f}  {kv}")

    out = {
        "source_file":         os.path.basename(source),
        "n_pairs":             n_pairs,
        "architecture":        arch,
        "clean_heads":         clean_heads,
        "total_heads":         total_heads,
        "delta_stats": {
            "mean": round(float(arr.mean()), 5),
            "std":  round(float(arr.std()), 5),
            "p90":  round(float(np.percentile(arr, 90)), 5),
            "p95":  round(float(np.percentile(arr, 95)), 5),
            "max":  round(float(arr.max()), 5),
            "min":  round(float(arr.min()), 5),
        },
        "retrieval_sweep":     ret_results,
        "induction_sweep":     ind_results,
        "verdict":             verdict,
        "conclusion":          conclusion,
    }
    out_path = os.path.join(OUT_DIR, "llama_diffuse_threshold.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
