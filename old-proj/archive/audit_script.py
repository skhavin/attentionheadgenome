import sys, os, pickle
import numpy as np

OUTPUT_DIR = "outputs/phase4"
LLAMA_PATTERNS = os.path.join(OUTPUT_DIR, "meta-llama-3.1-8b-bnb-4bit_attention_patterns.pkl")
LLAMA_PROTOTYPES = os.path.join(OUTPUT_DIR, "meta-llama-3.1-8b-bnb-4bit_prototypes.pkl")

def print_phi_vectors():
    print("=== 1. FULL PHI VECTORS (Sample) ===")
    if not os.path.exists(LLAMA_PATTERNS):
        print("Patterns file not found.")
        return
    with open(LLAMA_PATTERNS, "rb") as f:
        patterns = pickle.load(f)
    print(f"Loaded {len(patterns)} documents for LLaMA 3.1 8B.")
    
    # Just print Layer 0 Head 0 for first 3 docs to avoid flooding
    print("Sample Layer 0, Head 0 phi vectors (sink, local, entropy):")
    for i in range(min(3, len(patterns))):
        if (0, 0) in patterns[i]:
            print(f"  Doc {i}: {patterns[i][(0, 0)]}")
            
    if os.path.exists(LLAMA_PROTOTYPES):
        with open(LLAMA_PROTOTYPES, "rb") as f:
            prototypes = pickle.load(f)
        if (0, 0) in prototypes:
            print(f"\nK-Means Centroids for Layer 0 Head 0:")
            print(prototypes[(0, 0)]["centroids"])
            print(f"K-Means Labels (first 10 docs):")
            print(prototypes[(0, 0)]["labels"][:10])

def check_streaming_llm():
    print("\n=== 2. STREAMING LLM INDEXING LOGIC ===")
    def streaming_llm_indices(seq_len, budget):
        sink_count = min(4, budget)
        recent_count = budget - sink_count
        sinks = list(range(sink_count))
        recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
        return sorted(set(sinks + recents))[:budget]
    
    budget = 512
    for seq_len in [384, 448, 512, 576, 640]:
        idx = streaming_llm_indices(seq_len, budget)
        print(f"SeqLen={seq_len:3d}, Budget={budget:3d} => Indices Selected={len(idx)}, Sinks=[{min(idx)}...{idx[min(3, len(idx)-1)]}], Recents=[{idx[4] if len(idx)>4 else ''}...{max(idx)}]")

if __name__ == "__main__":
    print_phi_vectors()
    check_streaming_llm()
