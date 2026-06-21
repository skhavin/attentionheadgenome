import pickle
import numpy as np
import os
from sklearn.cluster import KMeans

STABILITY_DOC_COUNTS = [100, 300, 500]
NUM_CLUSTERS = 4

def check_stability(file_path, model_name):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"\nEvaluating stability for {model_name}...")
    with open(file_path, "rb") as f:
        all_patterns = pickle.load(f)

    # Need at least 500 docs
    if len(all_patterns) < 500:
        print(f"Warning: only found {len(all_patterns)} docs for {model_name}.")
    
    keys = sorted(list(all_patterns[0].keys()))[:10]  # Just take a subset of heads to speed up evaluation

    drifts = []
    prev_centroids = {}

    for n_docs in STABILITY_DOC_COUNTS:
        subset = all_patterns[:n_docs]
        if len(subset) < n_docs:
            print(f"  Only {len(subset)} docs available, skipping n_docs={n_docs}")
            continue

        curr_centroids = {}
        for layer, head in keys:
            histograms = [d[(layer, head)] for d in subset if (layer, head) in d]
            data = np.array(histograms)
            if len(data) == 0:
                continue
            k = min(NUM_CLUSTERS, len(data))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(data)
            curr_centroids[(layer, head)] = kmeans.cluster_centers_

        if prev_centroids:
            total_drift = 0
            count = 0
            for key in keys:
                if key in prev_centroids and key in curr_centroids:
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

def main():
    files = [
        ("GPT-2", "outputs/phase1/attention_patterns.pkl"),
        ("Qwen", "outputs/phase4/qwen_attention_patterns.pkl"),
        ("LLaMA", "outputs/phase4/meta-llama-3.1-8b-bnb-4bit_attention_patterns.pkl"),
    ]
    
    for name, path in files:
        check_stability(path, name)

if __name__ == "__main__":
    main()
