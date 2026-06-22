# step5_negative_control.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Prove that attention head clusters are LEARNED, not geometric artifacts
#          of causal masking + softmax.
#
# METHOD:
#   1. Build a GPT-2 Medium with RANDOM weights (no pretrained checkpoint).
#   2. Run the exact same profiling pipeline on the same 300 shared docs.
#   3. Cluster all heads together (k=4) based on their mean attention histograms.
#   4. Compare inertia and silhouette vs. the trained GPT-2.
#
# EXPECTED RESULT:
#   Random weights  -> flat, indistinct clusters, high inertia, low silhouette.
#   Trained model   -> sharp, well-separated clusters, low inertia, high silhouette.
#
# OUTPUTS:
#   outputs/phase1/negative_control.pkl
#   outputs/phase1/negative_control.json

import os
import sys
import json
import pickle
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "outputs", "phase1")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.json")
TRAINED_PKL= os.path.join(OUT_DIR, "gpt2-medium_patterns.pkl")
OUT_PKL    = os.path.join(OUT_DIR, "negative_control.pkl")
OUT_JSON   = os.path.join(OUT_DIR, "negative_control.json")

K_CLUSTERS = 4
NUM_DOCS   = 300   # same as the trained run


def compute_cluster_stats(all_patterns):
    """
    K-means cluster the mean histograms of all heads pooled together.
    Returns (inertia, silhouette) across all heads.
    """
    if not all_patterns:
        return None, None

    keys = sorted(all_patterns[0].keys())

    # Compute mean histogram for each head
    head_means = []
    for layer, head in keys:
        hists = np.array([d[(layer, head)] for d in all_patterns if (layer, head) in d])
        if len(hists) == 0:
            continue
        head_means.append(hists.mean(axis=0))

    if not head_means:
        return None, None

    X = np.array(head_means) # (num_heads, seq_len)

    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    inertia = float(km.inertia_)
    sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else 0.0
    return inertia, sil


def main():
    if not os.path.exists(INDEX_PATH):
        print("[ERROR] dataset_index.json missing. Run step1_generate_index.py first.")
        sys.exit(1)
    if not os.path.exists(TRAINED_PKL):
        print("[ERROR] gpt2-medium_patterns.pkl missing. Run step3_profile_gpt2.py first.")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from step3_profile_shared import profile

    # ── load trained GPT-2 stats ──────────────────────────────────────────
    print("Loading trained GPT-2 patterns...")
    with open(TRAINED_PKL, "rb") as f:
        trained_patterns = pickle.load(f)
    trained_inertia, trained_sil = compute_cluster_stats(trained_patterns)
    print(f"  Trained  -> inertia={trained_inertia:.4f}  silhouette={trained_sil:.4f}")

    # We can load the random control patterns saved in the previous run to save time
    random_pkl = os.path.join(OUT_DIR, "gpt2-random-control_patterns.pkl")
    if os.path.exists(random_pkl):
        print("Loading cached random GPT-2 patterns...")
        with open(random_pkl, "rb") as f:
            random_patterns = pickle.load(f)
    else:
        # ── build random-weight GPT-2 ─────────────────────────────────────────
        print("\nBuilding random-weight GPT-2 Medium (no pretrained checkpoint)...")
        config = AutoConfig.from_pretrained("gpt2-medium")
        config.output_attentions = True
        model  = AutoModelForCausalLM.from_config(config)   # random weights
        tok    = AutoTokenizer.from_pretrained("gpt2-medium")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if torch.cuda.is_available():
            model = model.half().cuda()
        model.eval()

        # ── profile random model ───────────────────────────────────────────────
        print("Profiling random-weight model on 300 shared docs...")
        random_patterns = profile(model, tok, INDEX_PATH, NUM_DOCS, OUT_DIR, "gpt2-random-control")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    random_inertia, random_sil = compute_cluster_stats(random_patterns)
    print(f"  Random   -> inertia={random_inertia:.4f}  silhouette={random_sil:.4f}")

    # ── verdict ───────────────────────────────────────────────────────────
    # Pass: random silhouette is significantly lower and random inertia is comparable or higher
    # In practice, silhouette difference is the most robust signal for clustering quality.
    verdict = "PASS" if (trained_sil > 1.5 * random_sil) else "FAIL"
    note = (
        "Clusters are learned circuit behaviors, not softmax geometry artifacts."
        if verdict == "PASS"
        else "WARNING: Random weights produce similar clusters. May be softmax artifact."
    )

    out = {
        "trained_gpt2": {
            "avg_inertia":    round(trained_inertia, 4) if trained_inertia else None,
            "avg_silhouette": round(trained_sil, 4)     if trained_sil     else None,
        },
        "random_gpt2": {
            "avg_inertia":    round(random_inertia, 4) if random_inertia else None,
            "avg_silhouette": round(random_sil, 4)     if random_sil     else None,
        },
        "verdict": verdict,
        "note":    note,
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*50}")
    print(f"VERDICT: {verdict}")
    print(f"NOTE:    {note}")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
