import json
import os
import torch
import numpy as np
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_dtw(seq1, seq2):
    """
    Computes DTW cost (Euclidean distance) between two trajectory matrices.
    seq1: [L1, D], seq2: [L2, D]
    """
    L1 = seq1.shape[0]
    L2 = seq2.shape[0]
    
    # Compute pairwise Euclidean distance between all temporal steps
    dist = torch.cdist(seq1.unsqueeze(0), seq2.unsqueeze(0)).squeeze(0).tolist() # [L1, L2] python list
    
    dtw = [[float('inf')] * (L2 + 1) for _ in range(L1 + 1)]
    dtw[0][0] = 0.0
    
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            cost = dist[i-1][j-1]
            dtw[i][j] = cost + min(
                dtw[i-1][j],    # insertion
                dtw[i][j-1],    # deletion
                dtw[i-1][j-1]   # match
            )
            
    return dtw[L1][L2]

def load_data():
    with open("../outputs/dataset/trajectory_mapping.json", "r") as f:
        mapping_prompts = json.load(f)
        
    categories = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    cat_indices = {cat: [] for cat in categories}
    for i, p in enumerate(mapping_prompts):
        cat_indices[p["task_type"]].append(i)
        
    out_dir = "../outputs/trajectories"
    models = ["Qwen2.5-1.5B", "Llama-3.2-1B", "phi-1_5"]
    
    data = {}
    for m in models:
        path = os.path.join(out_dir, m, "deconfounded_trajectories.pt")
        # Load and keep on CPU to save VRAM, we'll push to GPU dynamically if needed
        data[m] = torch.load(path, map_location="cpu")
        
    return cat_indices, data, categories, models

def compute_centroids(trajectories, indices):
    # trajectories: [N, L, D]
    # indices: list of prompt indices
    # Returns: [L, D] L2-normalized
    mean_vec = trajectories[indices].mean(dim=0)
    norm = mean_vec.norm(dim=1, keepdim=True) + 1e-8
    return mean_vec / norm

def run_bootstrap_ci(data_A, data_B, cat_indices, categories, n_bootstrap=1000):
    diagonal_costs = []
    off_diagonal_costs = []
    
    print(f"Running {n_bootstrap} bootstrap iterations...")
    start_time = time.time()
    
    for b in range(n_bootstrap):
        if b > 0 and b % 100 == 0:
            print(f"  Bootstrap {b}/{n_bootstrap} ({(time.time()-start_time):.1f}s)")
            
        centroids_A = {}
        centroids_B = {}
        
        # Compute resampled full-depth L2-normalized centroids for all categories
        raw_centroids_A = {}
        raw_centroids_B = {}
        for cat in categories:
            idx = cat_indices[cat]
            resampled_idx_A = np.random.choice(idx, size=len(idx), replace=True).tolist()
            resampled_idx_B = np.random.choice(idx, size=len(idx), replace=True).tolist()
            
            raw_centroids_A[cat] = compute_centroids(data_A, resampled_idx_A).to(DEVICE)
            raw_centroids_B[cat] = compute_centroids(data_B, resampled_idx_B).to(DEVICE)
        
        # Build Anchor Matrices from the FINAL layer (L-1)
        # shape: [6, D]
        anchor_A = torch.stack([raw_centroids_A[c][-1, :] for c in categories]) 
        anchor_B = torch.stack([raw_centroids_B[c][-1, :] for c in categories])
        
        # Project full trajectories into 6D Anchor Space
        centroids_A = {c: raw_centroids_A[c] @ anchor_A.T for c in categories}
        centroids_B = {c: raw_centroids_B[c] @ anchor_B.T for c in categories}
            
        # Compute DTW costs in the shared 6D space
        b_diag = []
        b_off_diag = []
        
        for c1 in categories:
            for c2 in categories:
                cost = compute_dtw(centroids_A[c1], centroids_B[c2])
                if c1 == c2:
                    b_diag.append(cost)
                else:
                    b_off_diag.append(cost)
                    
        diagonal_costs.append(np.mean(b_diag))
        off_diagonal_costs.append(np.mean(b_off_diag))
        
    diag_ci = np.percentile(diagonal_costs, [2.5, 97.5])
    off_diag_ci = np.percentile(off_diagonal_costs, [2.5, 97.5])
    
    return np.mean(diagonal_costs), diag_ci, np.mean(off_diagonal_costs), off_diag_ci

def main():
    cat_indices, data, categories, models = load_data()
    
    results = {}
    
    # 3 pairwise comparisons
    pairs = [
        ("Qwen2.5-1.5B", "Llama-3.2-1B"),
        ("Qwen2.5-1.5B", "phi-1_5"),
        ("Llama-3.2-1B", "phi-1_5")
    ]
    
    for mA, mB in pairs:
        print(f"\n{'='*50}\nAligning {mA} vs {mB}\n{'='*50}")
        
        # 1. Compute true 6x6 Matrix
        # Raw D-dimensional centroids
        raw_centroids_A = {cat: compute_centroids(data[mA], cat_indices[cat]).to(DEVICE) for cat in categories}
        raw_centroids_B = {cat: compute_centroids(data[mB], cat_indices[cat]).to(DEVICE) for cat in categories}
        
        # Final layer anchors: [6, D]
        anchor_A = torch.stack([raw_centroids_A[c][-1, :] for c in categories])
        anchor_B = torch.stack([raw_centroids_B[c][-1, :] for c in categories])
        
        # 6D Projections
        centroids_A = {c: raw_centroids_A[c] @ anchor_A.T for c in categories}
        centroids_B = {c: raw_centroids_B[c] @ anchor_B.T for c in categories}
        
        true_matrix = {}
        for c1 in categories:
            true_matrix[c1] = {}
            for c2 in categories:
                true_matrix[c1][c2] = compute_dtw(centroids_A[c1], centroids_B[c2])
                
        # 2. Run Bootstrap for Significance
        diag_mean, diag_ci, off_diag_mean, off_diag_ci = run_bootstrap_ci(
            data[mA], data[mB], cat_indices, categories, n_bootstrap=1000
        )
        
        print(f"\nResults for {mA} vs {mB}:")
        print(f"  Diagonal (Matched) DTW Cost: {diag_mean:.4f}  95% CI: [{diag_ci[0]:.4f}, {diag_ci[1]:.4f}]")
        print(f"  Off-Diagonal (Mismatched) Cost: {off_diag_mean:.4f}  95% CI: [{off_diag_ci[0]:.4f}, {off_diag_ci[1]:.4f}]")
        
        overlap = diag_ci[1] >= off_diag_ci[0]
        print(f"  Is Separation Significant (Non-overlapping 95% CIs)? {'No' if overlap else 'YES'}")
        
        results[f"{mA}_vs_{mB}"] = {
            "true_matrix": true_matrix,
            "diagonal_ci": list(diag_ci),
            "off_diagonal_ci": list(off_diag_ci),
            "significant": not overlap
        }
        
    os.makedirs("../outputs/dtw_results", exist_ok=True)
    with open("../outputs/dtw_results/dtw_cross_arch_alignment.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nAll cross-architecture DTW alignments successfully computed and saved!")

if __name__ == "__main__":
    main()
