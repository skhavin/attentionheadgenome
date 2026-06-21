# Cluster attention patterns per (layer, head) using K-Means.
# Input: attention_patterns.pkl from run_profiling.py
# Output: prototypes.pkl — dict of (layer, head) -> cluster centroids

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
from sklearn.cluster import KMeans
from config import PHASE1_DIR, NUM_CLUSTERS, PROTOTYPES_PATH

def main():
    # Load patterns
    patterns_path = os.path.join(PHASE1_DIR, "attention_patterns.pkl")
    with open(patterns_path, "rb") as f:
        all_patterns = pickle.load(f)
    print(f"Loaded patterns from {len(all_patterns)} documents")

    # Figure out all (layer, head) keys from first doc
    keys = sorted(all_patterns[0].keys())
    print(f"Found {len(keys)} (layer, head) pairs")

    prototypes = {}
    for layer, head in keys:
        # Gather this head's histogram from all docs
        histograms = []
        for doc_patterns in all_patterns:
            if (layer, head) in doc_patterns:
                histograms.append(doc_patterns[(layer, head)])

        data = np.array(histograms)  # (num_docs, MAX_SEQ_LEN)

        # Cluster
        k = min(NUM_CLUSTERS, len(data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)

        prototypes[(layer, head)] = {
            "centroids": kmeans.cluster_centers_,   # (k, MAX_SEQ_LEN)
            "labels": kmeans.labels_,               # (num_docs,)
            "inertia": kmeans.inertia_,
        }
        print(f"  Layer {layer:2d}, Head {head:2d}: {k} clusters, inertia={kmeans.inertia_:.4f}")

    with open(PROTOTYPES_PATH, "wb") as f:
        pickle.dump(prototypes, f)
    print(f"\nSaved prototypes to {PROTOTYPES_PATH}")

if __name__ == "__main__":
    main()
