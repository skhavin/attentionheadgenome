# Check if clusters stabilize as we use more documents.
# Re-cluster with 100, 300, 500 docs and measure centroid drift.
# Output: stability plot in outputs/phase1/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from config import PHASE1_DIR, NUM_CLUSTERS, STABILITY_DOC_COUNTS

def main():
    patterns_path = os.path.join(PHASE1_DIR, "attention_patterns.pkl")
    with open(patterns_path, "rb") as f:
        all_patterns = pickle.load(f)

    keys = sorted(all_patterns[0].keys())

    # For each doc count, cluster and record centroids
    # Then measure drift = average distance between successive centroid sets
    drifts = []
    prev_centroids = {}

    for n_docs in STABILITY_DOC_COUNTS:
        subset = all_patterns[:n_docs]
        curr_centroids = {}
        for layer, head in keys:
            histograms = [d[(layer, head)] for d in subset if (layer, head) in d]
            data = np.array(histograms)
            k = min(NUM_CLUSTERS, len(data))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            curr_centroids[(layer, head)] = kmeans.cluster_centers_

        # Measure drift from previous
        if prev_centroids:
            total_drift = 0
            count = 0
            for key in keys:
                if key in prev_centroids and key in curr_centroids:
                    # Match centroids by nearest neighbor, compute avg distance
                    prev = prev_centroids[key]
                    curr = curr_centroids[key]
                    for c in curr:
                        dists = np.linalg.norm(prev - c, axis=1)
                        total_drift += dists.min()
                        count += 1
            avg_drift = total_drift / max(count, 1)
            drifts.append(avg_drift)
        else:
            drifts.append(None)

        prev_centroids = curr_centroids
        print(f"  n_docs={n_docs}: drift={drifts[-1]}")

    # Plot
    valid_counts = [c for c, d in zip(STABILITY_DOC_COUNTS, drifts) if d is not None]
    valid_drifts = [d for d in drifts if d is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(valid_counts, valid_drifts, "o-", linewidth=2, markersize=8)
    plt.xlabel("Number of Documents")
    plt.ylabel("Average Centroid Drift")
    plt.title("Prototype Stability: Do Clusters Converge?")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(PHASE1_DIR, "stability_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved stability plot to {save_path}")

if __name__ == "__main__":
    main()
