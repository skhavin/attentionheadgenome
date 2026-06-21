# step6_characterize_cluster4.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Empirically inspect the 4th k-means centroid across all models.
#          Do NOT pre-label it. Assign a label only after inspecting its shape.
#
# VALID LABELS:
#   "induction"              -- shifted token copying, tracks [A][B]...[A]->[B] pattern
#   "composition"            -- structured position offsets into Q/K
#   "diffuse_background_noise" -- no distinguishing structure; uniform spread
#   "wide_local"             -- consistently attends to a broad but near window (not BOS)
#
# METHOD:
#   1. Load each model's pkl patterns.
#   2. Cluster with k=4 per (layer, head).
#   3. Identify which centroid is "cluster 4" by elimination:
#      - centroid with highest hist[0:4] = sink candidate
#      - centroid with highest hist[1:10] = local candidate
#      - centroid with highest std = retrieval candidate
#      - remaining centroid = cluster 4 (unknown)
#   4. Inspect its mean histogram shape. Measure peak position, breadth, symmetry.
#   5. Assign the appropriate label.
#
# OUTPUTS:
#   outputs/phase1/cluster_characterization.pkl
#   outputs/phase1/cluster_characterization.json

import os
import sys
import json
import pickle
import numpy as np
from sklearn.cluster import KMeans

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")
OUT_PKL  = os.path.join(OUT_DIR, "cluster_characterization.pkl")
OUT_JSON = os.path.join(OUT_DIR, "cluster_characterization.json")

K_CLUSTERS = 4

MODEL_SLUGS = {
    "GPT-2":     "gpt2-medium",
    "Qwen-0.5B": "qwen2.5-0.5b",
    "Qwen-1.5B": "qwen2.5-1.5b",
    "Llama-8B":  "llama-3.1-8b-4bit",
}

VALID_LABELS = [
    "induction",
    "composition",
    "diffuse_background_noise",
    "wide_local",
]


def identify_cluster4(centroids):
    """
    Given k=4 centroids (shape: (4, seq_len)), identify which one is NOT
    a clear sink, local, or retrieval prototype.

    Returns (cluster4_idx, annotations) where annotations describes each centroid.
    """
    n = centroids.shape[0]
    scores = []
    for i in range(n):
        c = centroids[i]
        scores.append({
            "idx":         i,
            "sink_score":  float(c[0:4].sum()),       # BOS mass
            "local_score": float(c[1:10].sum()),       # adjacent mass
            "ret_score":   float(c.std()),             # selectivity (proxy)
        })

    # Assign roles greedily to the centroid with highest score
    assigned = set()
    roles    = {}

    for role, field in [("sink", "sink_score"), ("local", "local_score"), ("retrieval", "ret_score")]:
        best = max((s for s in scores if s["idx"] not in assigned), key=lambda x: x[field])
        roles[role] = best["idx"]
        assigned.add(best["idx"])

    remaining = [i for i in range(n) if i not in assigned]
    cluster4_idx = remaining[0] if remaining else -1

    return cluster4_idx, roles, scores


def characterize_centroid(centroid):
    """
    Measure shape properties of a centroid histogram.
    Returns a dict of descriptors.
    """
    c = centroid
    peak_pos      = int(np.argmax(c))
    peak_val      = float(c[peak_pos])
    # Effective width: positions where hist > 10% of peak value
    above_thresh  = np.where(c > 0.1 * peak_val)[0]
    width         = int(above_thresh[-1] - above_thresh[0]) if len(above_thresh) > 1 else 0
    entropy       = float(-np.sum(c * np.log(c + 1e-12)))
    bos_mass      = float(c[0:4].sum())
    local_mass    = float(c[1:10].sum())
    mid_mass      = float(c[10:100].sum())
    far_mass      = float(c[100:].sum())

    return {
        "peak_position":   peak_pos,
        "peak_value":      round(peak_val, 4),
        "effective_width": width,
        "entropy":         round(entropy, 4),
        "bos_mass_0_3":    round(bos_mass, 4),
        "local_mass_1_9":  round(local_mass, 4),
        "mid_mass_10_99":  round(mid_mass, 4),
        "far_mass_100+":   round(far_mass, 4),
    }


def infer_label(desc):
    """
    Rule-based label assignment from centroid descriptor.
    These rules are empirical — inspect the output and override if needed.
    """
    # Induction: strong mid-range attention with some local component
    if desc["mid_mass_10_99"] > 0.3 and desc["local_mass_1_9"] > 0.15:
        return "induction", "Mid-range + local mass coexist — induction/composition pattern"
    # Wide local: peak in [1-30] but wide spread
    if desc["peak_position"] < 30 and desc["effective_width"] > 20:
        return "wide_local", "Broad attention centered near current token"
    # Diffuse: high entropy, no clear peak
    if desc["entropy"] > 5.0:
        return "diffuse_background_noise", "High entropy, no dominant structure"
    # Composition: narrow mid-range peak
    if 10 < desc["peak_position"] < 80 and desc["effective_width"] < 30:
        return "composition", "Narrow mid-range peak — possible Q/K composition"
    return "diffuse_background_noise", "No clear mechanistic signature identified"


def main():
    results = {}

    for model_name, slug in MODEL_SLUGS.items():
        pkl_path = os.path.join(OUT_DIR, f"{slug}_patterns.pkl")
        if not os.path.exists(pkl_path):
            print(f"[SKIP] {model_name} — patterns pkl not found.")
            results[model_name] = {"status": "missing"}
            continue

        print(f"\n[{model_name}] Loading patterns...")
        with open(pkl_path, "rb") as f:
            all_patterns = pickle.load(f)

        keys = sorted(all_patterns[0].keys())

        # Cluster all docs for each (layer, head) and collect centroids
        all_centroids = []
        for layer, head in keys:
            hists = np.array([d[(layer, head)] for d in all_patterns if (layer, head) in d])
            k     = min(K_CLUSTERS, len(hists))
            km    = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(hists)
            all_centroids.append(km.cluster_centers_)

        # Pool centroids across all heads, re-cluster to find global k=4 archetypes
        pooled = np.vstack(all_centroids)   # (n_heads * k, seq_len)
        km_global = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
        km_global.fit(pooled)
        global_centroids = km_global.cluster_centers_   # (4, seq_len)

        cluster4_idx, roles, scores = identify_cluster4(global_centroids)
        centroid4 = global_centroids[cluster4_idx] if cluster4_idx >= 0 else None

        if centroid4 is not None:
            desc  = characterize_centroid(centroid4)
            label, reason = infer_label(desc)
        else:
            desc   = {}
            label  = "indeterminate"
            reason = "Could not identify 4th centroid"

        print(f"  Centroid roles: sink={roles.get('sink')}, local={roles.get('local')}, retrieval={roles.get('retrieval')}, cluster4={cluster4_idx}")
        print(f"  Cluster 4 descriptor: {desc}")
        print(f"  => Assigned label: [{label}]  Reason: {reason}")

        results[model_name] = {
            "cluster4_centroid_idx": cluster4_idx,
            "centroid_role_map":     roles,
            "cluster4_descriptor":  desc,
            "inferred_label":        label,
            "reason":                reason,
            "valid_labels":          VALID_LABELS,
            "note": (
                "Label assigned by automated heuristic. "
                "Inspect centroid shape manually and override if needed."
            ),
        }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {OUT_PKL}")
    print(f"Saved -> {OUT_JSON}")
    print("\n[REMINDER] These labels are heuristic. Open cluster_characterization.json")
    print("and manually verify or override the 'inferred_label' field before Phase 4.")


if __name__ == "__main__":
    main()
