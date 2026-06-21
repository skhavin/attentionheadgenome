import sys, os
import pickle
import numpy as np
from sklearn.cluster import KMeans

# We define extreme descriptors:
# Local: mass in relative positions [0, 10]
# Sink: mass in relative positions [MAX_SEQ_LEN - 50, MAX_SEQ_LEN] 
#       (Since sink token is at position 0, relative distance to it is large)
# Retrieval: mass in [10, MAX_SEQ_LEN - 50]

MAX_SEQ_LEN = 512

MODELS = {
    "GPT-2": r"D:\PROJECTS\webstromprojects\supertransformers\outputs\phase1\attention_patterns.pkl",
    "Qwen-0.5B": r"D:\PROJECTS\webstromprojects\supertransformers\outputs\phase4\qwen_attention_patterns.pkl",
    "Llama-8B": r"D:\PROJECTS\webstromprojects\supertransformers\outputs\phase4\meta-llama-3.1-8b-bnb-4bit_attention_patterns.pkl"
}

def analyze_model(name, path):
    if not os.path.exists(path):
        print(f"Skipping {name}, path not found: {path}")
        return None

    with open(path, "rb") as f:
        all_patterns = pickle.load(f)

    keys = sorted(all_patterns[0].keys())
    total_layers = max(k[0] for k in keys) + 1

    head_descriptors = {}
    X = []
    head_list = []

    for layer, head in keys:
        histograms = [d[(layer, head)] for d in all_patterns if (layer, head) in d]
        if not histograms:
            continue
        avg_hist = np.mean(histograms, axis=0)
        # Normalize just in case
        if avg_hist.sum() > 0:
            avg_hist = avg_hist / avg_hist.sum()
        
        head_descriptors[(layer, head)] = avg_hist
        X.append(avg_hist)
        head_list.append((layer, head))

    X = np.array(X)
    
    # K-means clustering of the heads (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Compute extreme metrics
    extreme_scores = []
    for i, (layer, head) in enumerate(head_list):
        hist = X[i]
        local_mass = hist[:10].sum()
        sink_mass = hist[-50:].sum()
        retrieval_mass = hist[10:-50].sum()
        
        rel_depth = layer / max(1, total_layers - 1)
        
        extreme_scores.append({
            "layer": layer,
            "head": head,
            "rel_depth": rel_depth,
            "cluster": labels[i],
            "local": local_mass,
            "sink": sink_mass,
            "retrieval": retrieval_mass
        })

    # Sort to find top 5
    top_local = sorted(extreme_scores, key=lambda x: x["local"], reverse=True)[:5]
    top_sink = sorted(extreme_scores, key=lambda x: x["sink"], reverse=True)[:5]
    top_retrieval = sorted(extreme_scores, key=lambda x: x["retrieval"], reverse=True)[:5]

    return {
        "top_local": top_local,
        "top_sink": top_sink,
        "top_retrieval": top_retrieval,
        "total_layers": total_layers
    }

def main():
    results = {}
    for name, path in MODELS.items():
        print(f"\nAnalyzing {name}...")
        res = analyze_model(name, path)
        if res:
            results[name] = res

    print("\n" + "="*50)
    print("EXTREME HEADS COMPARISON ACROSS ARCHITECTURES")
    print("="*50)

    for category in ["local", "sink", "retrieval"]:
        print(f"\n--- Top 5 {category.upper()} Heads ---")
        for name, res in results.items():
            print(f"\nModel: {name} (Total Layers: {res['total_layers']})")
            for head_info in res[f"top_{category}"]:
                print(f"  Layer {head_info['layer']:2d}, Head {head_info['head']:2d} | "
                      f"Rel Depth: {head_info['rel_depth']:.3f} | "
                      f"Cluster: {head_info['cluster']} | "
                      f"Score ({category}): {head_info[category]:.4f}")

if __name__ == "__main__":
    main()
