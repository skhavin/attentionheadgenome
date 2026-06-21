# step4_extreme_heads.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Load all 4 model pattern summaries, compute corrected head-type scores,
#          extract top-5 extreme heads per type per model, and compare relative depths.
#
# CORRECTED METRICS (per plan-latest.md v3):
#   sink_score     = hist[0:4].sum()      -- BOS anchor mass (distance 0-3)
#   local_score    = hist[1:10].sum()     -- adjacent token mass (distance 1-9)
#   retrieval_score = entropy_delta       -- computed with a synthetic KV probe
#   rel_depth      = layer / (total_layers - 1)
#
# NOTE ON RETRIEVAL: True entropy-delta requires a live model forward pass.
#   This script uses a PROXY metric: std of the mean histogram (high std =
#   high selectivity = more retrieval-like). This is a fast, model-free approximation.
#   When models are loaded for re-profiling, the live entropy delta can replace this.
#
# OUTPUTS:
#   outputs/phase1/extreme_heads_comparison.pkl
#   outputs/phase1/extreme_heads_comparison.json

import os
import sys
import json
import pickle
import numpy as np
from sklearn.cluster import KMeans

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")

K_CLUSTERS = 4
TOP_N      = 5     # top-5 extreme heads per type

# ── which models to load and what their slugs are ─────────────────────────────
# Adjust slugs if you profiled with different names
MODEL_SLUGS = {
    "GPT-2":     "gpt2-medium",
    "Qwen-0.5B": "qwen2.5-0.5b",
    "Qwen-1.5B": "qwen2.5-1.5b",
    "Llama-8B":  "llama-3.1-8b-4bit",
}


def load_model_summary(out_dir, slug):
    """
    Load per-head mean histograms from json summary.
    Returns dict: (layer, head) -> np.ndarray shape (512,)
    Returns None if file doesn't exist.
    """
    json_path = os.path.join(out_dir, f"{slug}_patterns_summary.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    heads = {}
    for key, hist in data["heads"].items():
        layer, head = int(key.split("_")[0]), int(key.split("_")[1])
        heads[(layer, head)] = np.array(hist, dtype=np.float32)
    return heads, data.get("num_docs", "?")


def score_heads(head_hists):
    """
    Compute sink, local, retrieval scores for every (layer, head).
    Returns list of dicts with keys: layer, head, sink, local, retrieval, rel_depth, cluster
    """
    if not head_hists:
        return []

    keys         = sorted(head_hists.keys())
    total_layers = max(k[0] for k in keys) + 1

    # Run k=4 clustering on all heads to get cluster assignments
    X        = np.array([head_hists[k] for k in keys])
    km       = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels   = km.fit_predict(X)

    records = []
    for i, (layer, head) in enumerate(keys):
        hist = head_hists[(layer, head)]

        # Corrected sink: mass at distances 0-3 (BOS anchor)
        sink  = float(hist[0:4].sum())
        # Local: mass at distances 1-9 (adjacent tokens)
        local = float(hist[1:10].sum())
        # Retrieval proxy: std of histogram (selectivity measure)
        # High std = attention concentrates sharply somewhere = retrieval-like
        retrieval = float(hist.std())

        rel_depth = layer / max(total_layers - 1, 1)

        records.append({
            "layer":     layer,
            "head":      head,
            "rel_depth": round(rel_depth, 4),
            "cluster":   int(labels[i]),
            "sink":      round(sink, 4),
            "local":     round(local, 4),
            "retrieval": round(retrieval, 6),
        })

    return records, total_layers


def top_n(records, field, n=5):
    """Return top-n records sorted by field descending."""
    return sorted(records, key=lambda x: x[field], reverse=True)[:n]


def grade_verdict(results):
    """
    Compare relative depths of top-5 sink heads across all available models.
    Returns a string verdict.
    """
    all_depths = {}
    for model_name, res in results.items():
        if "top_sink" in res:
            depths = [h["rel_depth"] for h in res["top_sink"]]
            all_depths[model_name] = depths

    if len(all_depths) < 2:
        return "Insufficient data — need >= 2 models"

    # Check if all top-sink heads fall within +-0.05 of a common band
    all_flat  = [d for depths in all_depths.values() for d in depths]
    band_min  = min(all_flat)
    band_max  = max(all_flat)
    band_size = band_max - band_min

    models_in_band = sum(
        1 for depths in all_depths.values()
        if max(depths) - min(depths) <= 0.10   # within 10% range
    )
    n_models = len(all_depths)

    if band_size <= 0.10 and models_in_band == n_models:
        return "Strong Genome (all models align within +-5% depth band)"
    elif models_in_band >= max(n_models - 1, 1):
        return "Scale-Dependent Genome (3/4 models align; one architecture diverges)"
    else:
        return "Null Hypothesis (depths scattered; no universal law found)"


def main():
    output = {
        "metadata": {
            "k_clusters":            K_CLUSTERS,
            "top_n":                 TOP_N,
            "sink_metric":           "hist[0:4].sum() -- BOS anchor distance 0-3",
            "local_metric":          "hist[1:10].sum() -- adjacent distance 1-9",
            "retrieval_metric":      "hist.std() -- proxy for selectivity (higher = more retrieval-like)",
            "retrieval_note":        "Proxy only. Replace with entropy-delta on live model when available.",
        }
    }

    any_loaded = False
    for model_name, slug in MODEL_SLUGS.items():
        result = load_model_summary(OUT_DIR, slug)
        if result is None:
            print(f"[SKIP] {model_name} ({slug}) — summary json not found, skipping.")
            output[model_name] = {"status": "missing"}
            continue

        head_hists, num_docs = result
        records, total_layers = score_heads(head_hists)
        any_loaded = True

        output[model_name] = {
            "total_layers": total_layers,
            "num_docs":     num_docs,
            "num_heads":    len(records),
            "top_sink":     top_n(records, "sink",      TOP_N),
            "top_local":    top_n(records, "local",     TOP_N),
            "top_retrieval":top_n(records, "retrieval", TOP_N),
        }
        print(f"\n[{model_name}]  layers={total_layers}  heads={len(records)}  docs={num_docs}")
        print(f"  Top-5 SINK      : {[(r['layer'], r['head'], r['rel_depth'], r['sink']) for r in output[model_name]['top_sink']]}")
        print(f"  Top-5 LOCAL     : {[(r['layer'], r['head'], r['rel_depth'], r['local']) for r in output[model_name]['top_local']]}")
        print(f"  Top-5 RETRIEVAL : {[(r['layer'], r['head'], r['rel_depth'], r['retrieval']) for r in output[model_name]['top_retrieval']]}")

    if not any_loaded:
        print("[ERROR] No model summaries found. Run step3_profile_*.py scripts first.")
        sys.exit(1)

    verdict = grade_verdict(output)
    output["verdict"] = verdict
    print(f"\n{'='*55}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*55}")

    # ── save pkl ──────────────────────────────────────────────────────────
    pkl_path = os.path.join(OUT_DIR, "extreme_heads_comparison.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(output, f)

    # ── save json ─────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "extreme_heads_comparison.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {pkl_path}")
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    main()
