# Compare prototype structures between GPT-2 and Qwen.
# Cluster Qwen patterns and compare cluster shapes to GPT-2's.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from config import PHASE1_DIR, PHASE4_DIR, NUM_CLUSTERS

def cluster_patterns(all_patterns, num_clusters):
    """Cluster attention patterns and return centroids per (layer, head)."""
    keys = sorted(all_patterns[0].keys())
    prototypes = {}
    for layer, head in keys:
        histograms = [d[(layer, head)] for d in all_patterns if (layer, head) in d]
        data = np.array(histograms)
        k = min(num_clusters, len(data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        prototypes[(layer, head)] = kmeans.cluster_centers_
    return prototypes

def main():
    # Load GPT-2 patterns
    with open(os.path.join(PHASE1_DIR, "attention_patterns.pkl"), "rb") as f:
        gpt2_patterns = pickle.load(f)

    # Load Qwen patterns
    with open(os.path.join(PHASE4_DIR, "qwen_attention_patterns.pkl"), "rb") as f:
        qwen_patterns = pickle.load(f)

    gpt2_proto = cluster_patterns(gpt2_patterns, NUM_CLUSTERS)
    qwen_proto = cluster_patterns(qwen_patterns, NUM_CLUSTERS)

    # Compare: for each model, compute the "locality ratio" of each head
    # (fraction of centroid weight in first 10 positions = local head)
    def locality_ratios(prototypes):
        ratios = []
        for key, centroids in sorted(prototypes.items()):
            for c in centroids:
                ratios.append(c[:10].sum())  # weight in first 10 relative positions
        return ratios

    gpt2_lr = locality_ratios(gpt2_proto)
    qwen_lr = locality_ratios(qwen_proto)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(gpt2_lr, bins=30, alpha=0.7, color="steelblue")
    axes[0].set_title("GPT-2 Medium: Prototype Locality Ratios")
    axes[0].set_xlabel("Locality (weight in positions 0-9)")
    axes[0].set_ylabel("Count")

    axes[1].hist(qwen_lr, bins=30, alpha=0.7, color="coral")
    axes[1].set_title("Qwen2.5-0.5B: Prototype Locality Ratios")
    axes[1].set_xlabel("Locality (weight in positions 0-9)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    save_path = os.path.join(PHASE4_DIR, "arch_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_path}")

    # Summary stats
    print(f"\nGPT-2 locality: mean={np.mean(gpt2_lr):.3f}, std={np.std(gpt2_lr):.3f}")
    print(f"Qwen  locality: mean={np.mean(qwen_lr):.3f}, std={np.std(qwen_lr):.3f}")

if __name__ == "__main__":
    main()
