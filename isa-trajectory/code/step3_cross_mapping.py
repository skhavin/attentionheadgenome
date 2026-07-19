import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def precompute_gram(T_seq):
    # T_seq: [L, 30, D]
    L = T_seq.shape[0]
    K_cs = []
    vars = []
    for u in range(L):
        X = T_seq[u].numpy()
        K = X @ X.T
        K_c = center(K)
        K_cs.append(K_c)
        vars.append(np.sqrt(np.sum(K_c * K_c)))
    return K_cs, vars

def fast_cka(K_c, varK, L_c, varL):
    hsic = np.sum(K_c * L_c)
    return hsic / (varK * varL + 1e-12)

def dtw_sakoe_chiba(D, radius):
    L1, L2 = D.shape
    cost = np.full((L1 + 1, L2 + 1), np.inf)
    cost[0, 0] = 0
    
    for i in range(1, L1 + 1):
        c = int(i * L2 / L1)
        start_j = max(1, c - radius)
        end_j = min(L2 + 1, c + radius + 1)
        
        for j in range(start_j, end_j):
            cost[i, j] = D[i-1, j-1] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            
    # Backtrack
    path = []
    i, j = L1, L2
    if np.isinf(cost[i, j]):
        return np.inf, []
        
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        if i == 1 and j == 1:
            break
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            choices = [cost[i-1, j-1], cost[i-1, j], cost[i, j-1]]
            m = np.argmin(choices)
            if m == 0:
                i -= 1; j -= 1
            elif m == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()
    return cost[L1, L2] / len(path), path

def build_distance_matrix_fast(K1s, varK1s, K2s, varK2s):
    L_A = len(K1s)
    L_B = len(K2s)
    D = np.zeros((L_A, L_B))
    
    for u in range(L_A):
        for v in range(L_B):
            D[u, v] = 1.0 - fast_cka(K1s[u], varK1s[u], K2s[v], varK2s[v])
    return D

def load_data():
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        test_prompts = json.load(f)
        
    categories = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    cat_indices = {c: [] for c in categories}
    for i, p in enumerate(test_prompts):
        cat_indices[p["task_type"]].append(i)
        
    models = ["Qwen2.5-1.5B", "Llama-3.2-1B", "phi-1_5"]
    T_dict = {}
    for m in models:
        path = f"../outputs/trajectories/{m}/val_deconfounded_trajectories.pt"
        T_dict[m] = torch.load(path, map_location="cpu")
        
    return models, categories, cat_indices, T_dict

def main():
    out_dir = "../outputs/cross_mapping"
    os.makedirs(out_dir, exist_ok=True)
    
    models, categories, cat_indices, T_dict = load_data()
    n_cat = len(categories)
    
    # Precompute all Gram matrices for all models and categories
    gram_dict = {}
    for m in models:
        gram_dict[m] = {}
        for c in categories:
            seq = T_dict[m][cat_indices[c]].permute(1, 0, 2)
            gram_dict[m][c] = precompute_gram(seq)
            
    pairs = [
        ("Qwen2.5-1.5B", "Llama-3.2-1B"),
        ("Qwen2.5-1.5B", "phi-1_5"),
        ("Llama-3.2-1B", "phi-1_5")
    ]
    
    dtw_radius = 6 # roughly 20-35% of layers
    n_shuffles = 100
    results_json = {}
    
    for m1, m2 in pairs:
        print(f"\n{'='*50}\nAligning {m1} and {m2}\n{'='*50}")
        
        confusion = np.zeros((n_cat, n_cat))
        paths = {}
        
        # 1. Build 6x6 confusion matrix
        print("Computing 6x6 cross-category trajectory alignments...")
        for i, c1 in enumerate(categories):
            K1s, varK1s = gram_dict[m1][c1]
            for j, c2 in enumerate(categories):
                K2s, varK2s = gram_dict[m2][c2]
                
                D_mat = build_distance_matrix_fast(K1s, varK1s, K2s, varK2s)
                cost, path = dtw_sakoe_chiba(D_mat, dtw_radius)
                
                confusion[i, j] = cost
                if i == j:
                    paths[c1] = (D_mat, path)
                    
        # 2. Time-Shuffle Control (on the diagonal matching categories)
        print("Computing Time-Shuffle Null Baseline...")
        shuffle_5ths = []
        for i, c1 in enumerate(categories):
            D_mat = paths[c1][0] # Retrieve precomputed distance matrix
            
            shuffle_costs = []
            for _ in range(n_shuffles):
                # Scramble layers of Model 2 (shuffle columns of D_mat)
                shuffled_idx = np.random.permutation(D_mat.shape[1])
                D_shuffled = D_mat[:, shuffled_idx]
                cost, _ = dtw_sakoe_chiba(D_shuffled, dtw_radius)
                if not np.isinf(cost):
                    shuffle_costs.append(cost)
                
            if len(shuffle_costs) > 0:
                shuffle_5ths.append(np.percentile(shuffle_costs, 5))
            
        avg_shuffle_threshold = np.mean(shuffle_5ths) if len(shuffle_5ths) > 0 else np.inf
        diag_mean = np.diag(confusion).mean()
        print(f"[{m1} vs {m2}] Mean True Diagonal Cost: {diag_mean:.4f}")
        print(f"[{m1} vs {m2}] 5th Percentile Time-Shuffle Threshold: {avg_shuffle_threshold:.4f} (Lower is better)")
        
        results_json[f"{m1}_vs_{m2}"] = {
            "confusion_matrix": confusion.tolist(),
            "mean_diagonal_cost": float(diag_mean),
            "time_shuffle_threshold": float(avg_shuffle_threshold)
        }
        
        # 3. Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt=".3f", cmap="viridis_r",
                    xticklabels=categories, yticklabels=categories)
        plt.title(f"Trajectory Alignment Cost (DTW-CKA)\n{m1} (Y) vs {m2} (X)")
        plt.ylabel(m1)
        plt.xlabel(m2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_{m1.replace('/', '_')}_{m2.replace('/', '_')}.png"), dpi=300)
        plt.close()
        
        # 4. Plot DTW Paths for the diagonal
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, c1 in enumerate(categories):
            ax = axes[i]
            if c1 not in paths: continue
            D_mat, path = paths[c1]
            if len(path) == 0: continue
            
            im = ax.imshow(D_mat, origin='lower', cmap='plasma', aspect='auto')
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, color='white', linewidth=3, marker='o', markersize=4)
            
            ax.set_title(f"{c1.capitalize()}\nCost: {confusion[i,i]:.3f}")
            ax.set_ylabel(f"{m1} Layers")
            ax.set_xlabel(f"{m2} Layers")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        plt.suptitle(f"Optimal Trajectory Alignment Paths\n{m1} vs {m2}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"paths_{m1.replace('/', '_')}_{m2.replace('/', '_')}.png"), dpi=300)
        plt.close()

    with open(os.path.join(out_dir, "alignment_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

if __name__ == "__main__":
    main()
