# step1_depth_trajectory.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Map attention head roles chronologically across relative depth bins
#          to validate the "Spatial Law" of attention head functional emergence.
#
# OUTPUTS:
#   outputs/phase2/depth_trajectory.json

import os
import sys
import json
import numpy as np
from sklearn.cluster import KMeans

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR     = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR    = os.path.join(ROOT, "outputs", "phase2")

MODEL_SLUGS = {
    "GPT-2":        "gpt2-medium",
    "Qwen-0.5B":    "qwen2.5-0.5b",
    "Qwen-1.5B":    "qwen2.5-1.5b",
    "Llama-3.2-1B": "llama-3.2-1b",
}

K_CLUSTERS = 4
BIN_WIDTH  = 0.05
BINS       = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)


def map_cluster_roles(centroids):
    """
    Map each of the 4 cluster indices to a semantic role:
    sink, local, retrieval, induction.
    Uses std and sink_mass to resolve overlaps.
    """
    n = centroids.shape[0]
    stds = [float(c.std()) for c in centroids]
    sink_masses = [float(c[0:4].sum()) for c in centroids]
    
    # 1. Induction has the absolute lowest std (flattest distribution)
    induction_idx = int(np.argmin(stds))
    
    # 2. Sink has the highest sink mass (early positions)
    sink_idx = int(np.argmax(sink_masses))
    if sink_idx == induction_idx:
        # Fallback: pick the second highest sink mass
        sorted_sink_indices = np.argsort(sink_masses)[::-1]
        sink_idx = int(sorted_sink_indices[1])
        
    # 3. Of the remaining two:
    #    - The one with higher std is retrieval
    #    - The one with lower std is local
    remaining = [i for i in range(n) if i not in (induction_idx, sink_idx)]
    if stds[remaining[0]] > stds[remaining[1]]:
        retrieval_idx = remaining[0]
        local_idx = remaining[1]
    else:
        retrieval_idx = remaining[1]
        local_idx = remaining[0]
        
    return {
        sink_idx: "sink",
        local_idx: "local",
        retrieval_idx: "retrieval",
        induction_idx: "induction"
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for model_name, slug in MODEL_SLUGS.items():
        json_path = os.path.join(IN_DIR, f"{slug}_patterns_summary.json")
        if not os.path.exists(json_path):
            print(f"[SKIP] {model_name} summary not found at {json_path}")
            continue

        print(f"\nProcessing {model_name} ({slug})...")
        with open(json_path) as f:
            data = json.load(f)

        heads = {}
        for key, hist in data["heads"].items():
            layer, head = map(int, key.split("_"))
            heads[(layer, head)] = np.array(hist, dtype=np.float32)

        keys         = sorted(heads.keys())
        total_layers = max(k[0] for k in keys) + 1
        X            = np.array([heads[k] for k in keys])

        # Run k=4 clustering on all heads
        km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        # Map cluster indices to semantic roles
        role_map = map_cluster_roles(centroids)
        print(f"  Cluster roles: {role_map}")

        # Bin heads by relative depth
        # We define bins from 0.0 to 1.0 with width 0.05
        num_bins = len(BINS) - 1
        bin_counts = {role: np.zeros(num_bins) for role in ["sink", "local", "retrieval", "induction"]}

        for i, (layer, head) in enumerate(keys):
            rel_depth = layer / max(total_layers - 1, 1)
            role = role_map[labels[i]]

            # Find which bin this relative depth falls into
            bin_idx = min(int(rel_depth / BIN_WIDTH), num_bins - 1)
            bin_counts[role][bin_idx] += 1

        # Normalize counts per bin to get proportions
        bin_proportions = {}
        for bin_idx in range(num_bins):
            total_in_bin = sum(bin_counts[r][bin_idx] for r in bin_counts)
            bin_start = round(BINS[bin_idx], 2)
            bin_end   = round(BINS[bin_idx+1], 2)
            bin_label = f"{bin_start}-{bin_end}"

            if total_in_bin > 0:
                bin_proportions[bin_label] = {
                    role: round(bin_counts[role][bin_idx] / total_in_bin, 4)
                    for role in bin_counts
                }
            else:
                bin_proportions[bin_label] = {role: 0.0 for role in bin_counts}

        results[model_name] = {
            "total_layers": total_layers,
            "total_heads":  len(keys),
            "trajectories": bin_proportions,
        }

        # Print quick summary of depth trajectory
        print("  Depth Trajectory Proportions (selected bins):")
        for bin_label in ["0.0-0.05", "0.1-0.15", "0.4-0.45", "0.8-0.85", "0.95-1.0"]:
            if bin_label in bin_proportions:
                props = bin_proportions[bin_label]
                print(f"    Bin {bin_label}: sink={props['sink']:.2f}, local={props['local']:.2f}, ret={props['retrieval']:.2f}, ind={props['induction']:.2f}")

    # Save to outputs/phase2/depth_trajectory.json
    out_json = os.path.join(OUT_DIR, "depth_trajectory.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {out_json}")
    print("[DONE] Chronological depth mapping complete.")


if __name__ == "__main__":
    main()
