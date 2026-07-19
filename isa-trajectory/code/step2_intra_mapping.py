import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import seaborn as sns

def load_data():
    with open("../outputs/dataset/trajectory_mapping.json", "r") as f:
        train_prompts = json.load(f)
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        test_prompts = json.load(f)
    
    categories = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    
    train_cat_indices = {c: [] for c in categories}
    for i, p in enumerate(train_prompts):
        train_cat_indices[p["task_type"]].append(i)
        
    test_cat_indices = {c: [] for c in categories}
    for i, p in enumerate(test_prompts):
        test_cat_indices[p["task_type"]].append(i)
        
    return categories, train_cat_indices, test_cat_indices

def compute_f_statistic(X, labels_idx, n_categories=6):
    """
    X: [N, D]
    labels_idx: list of lists, where labels_idx[c] contains the indices for category c
    """
    N = X.shape[0]
    mu_global = X.mean(dim=0)
    
    SSB = 0.0
    SSW = 0.0
    
    for c_idx in range(n_categories):
        idx = labels_idx[c_idx]
        n_c = len(idx)
        if n_c == 0: continue
        
        X_c = X[idx]
        mu_c = X_c.mean(dim=0)
        
        SSB += n_c * torch.sum((mu_c - mu_global)**2)
        SSW += torch.sum((X_c - mu_c)**2)
        
    df_B = n_categories - 1
    df_W = N - n_categories
    
    MSB = SSB / df_B
    MSW = SSW / df_W
    
    F = MSB / (MSW + 1e-12)
    return F.item()

def run_f_stat_analysis(test_T, test_cat_indices, categories, n_shuffles=100):
    N, L, D = test_T.shape
    
    real_F = np.zeros(L)
    shuffle_95th = np.zeros(L)
    
    cat_list = list(test_cat_indices.keys())
    # flatten label assignments for easy shuffling
    labels = np.zeros(N, dtype=int)
    for c_idx, cat in enumerate(cat_list):
        labels[test_cat_indices[cat]] = c_idx
        
    print("Computing F-statistics and Shuffle Controls...")
    for l in range(L):
        X = test_T[:, l, :]
        
        # Real F
        real_F[l] = compute_f_statistic(X, [np.where(labels == c)[0] for c in range(6)])
        
        # Shuffle F
        shuffle_F_vals = []
        for _ in range(n_shuffles):
            shuffled_labels = np.random.permutation(labels)
            shuffled_idx = [np.where(shuffled_labels == c)[0] for c in range(6)]
            shuffle_F_vals.append(compute_f_statistic(X, shuffled_idx))
            
        shuffle_95th[l] = np.percentile(shuffle_F_vals, 95)
        
    return real_F, shuffle_95th

def plot_f_statistics(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (m_name, m_res) in enumerate(results.items()):
        ax = axes[i]
        real_F = m_res["real_F"]
        shuffle_95th = m_res["shuffle_95th"]
        L = len(real_F)
        layers = np.arange(L)
        
        ax.plot(layers, real_F, label="Real Divergence (F-ratio)", color="blue", linewidth=2.5)
        ax.plot(layers, shuffle_95th, label="95th %ile Shuffle", color="red", linestyle="--", linewidth=2)
        
        ax.set_title(m_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Layer Depth", fontsize=12)
        ax.set_ylabel("F-Statistic (SSB/SSW)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f_statistic_divergence.png"), dpi=300)
    plt.close()

def compute_centroids(T, cat_indices, categories):
    N, L, D = T.shape
    centroids = torch.zeros(6, L, D)
    for c_idx, cat in enumerate(categories):
        idx = cat_indices[cat]
        centroids[c_idx] = T[idx].mean(dim=0)
    return centroids

def run_pca_and_plot(train_T, test_centroids, categories, m_name, out_dir):
    N_train, L, D = train_T.shape
    
    print(f"[{m_name}] Fitting Global PCA on Train Manifold ({N_train * L} points, D={D})...")
    X_train_flat = train_T.reshape(-1, D).numpy()
    
    pca = PCA(n_components=2)
    pca.fit(X_train_flat)
    
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"[{m_name}] PCA 2-Component Explained Variance: {var_explained:.2f}%")
    
    # Project Test Centroids
    test_centroids_flat = test_centroids.reshape(6 * L, D).numpy()
    test_pca_flat = pca.transform(test_centroids_flat)
    test_pca = test_pca_flat.reshape(6, L, 2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = list(mcolors.TABLEAU_COLORS.values())[:6]
    
    for c_idx, cat in enumerate(categories):
        traj = test_pca[c_idx]
        # Plot trajectory line
        ax.plot(traj[:, 0], traj[:, 1], color=colors[c_idx], label=cat, linewidth=2, alpha=0.8)
        
        # Mark start (L0) and end (LN)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[c_idx], marker='o', s=100, edgecolor='black', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[c_idx], marker='*', s=200, edgecolor='black', zorder=5)
        
        # Add small text for LN to clarify direction
        ax.text(traj[-1, 0], traj[-1, 1], f" L{L-1}", fontsize=9)
        
    ax.set_title(f"{m_name} Test Trajectory Manifold\nGlobal PCA (Explained Variance: {var_explained:.1f}%)", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=cat) for i, cat in enumerate(categories)]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Start (Layer 0)', markerfacecolor='gray', markersize=10, markeredgecolor='black'))
    legend_elements.append(Line2D([0], [0], marker='*', color='w', label='End (Final Layer)', markerfacecolor='gray', markersize=15, markeredgecolor='black'))
    
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_trajectory_{m_name.replace('/', '_')}.png"), dpi=300)
    plt.close()
    
    return var_explained

def main():
    models = [
        "Qwen2.5-1.5B",
        "Llama-3.2-1B",
        "phi-1_5"
    ]
    
    out_dir = "../outputs/intra_mapping"
    os.makedirs(out_dir, exist_ok=True)
    
    categories, train_cat_indices, test_cat_indices = load_data()
    
    f_results = {}
    pca_stats = {}
    
    for m in models:
        print(f"\n{'='*50}\nAnalyzing {m}\n{'='*50}")
        
        train_path = f"../outputs/trajectories/{m}/deconfounded_trajectories.pt"
        test_path = f"../outputs/trajectories/{m}/val_deconfounded_trajectories.pt"
        
        if not os.path.exists(test_path):
            print(f"ERROR: {test_path} not found. Ensure extraction completed.")
            continue
            
        train_T = torch.load(train_path, map_location="cpu")
        test_T = torch.load(test_path, map_location="cpu")
        
        # 1. Divergence Dynamics (F-Statistic)
        real_F, shuffle_95th = run_f_stat_analysis(test_T, test_cat_indices, categories)
        f_results[m] = {"real_F": real_F, "shuffle_95th": shuffle_95th}
        
        # 2. PCA Manifold Projection
        test_centroids = compute_centroids(test_T, test_cat_indices, categories)
        var_explained = run_pca_and_plot(train_T, test_centroids, categories, m, out_dir)
        pca_stats[m] = var_explained
        
    if len(f_results) == 3:
        plot_f_statistics(f_results, out_dir)
        
        # Save exact F-stat data for report
        with open(os.path.join(out_dir, "f_statistic_data.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_results = {
                k: {
                    "real_F": v["real_F"].tolist(), 
                    "shuffle_95th": v["shuffle_95th"].tolist(),
                    "pca_explained_variance_2d": pca_stats[k]
                } 
                for k, v in f_results.items()
            }
            json.dump(json_safe_results, f, indent=2)
            
        print("\nAll Intra-Model Mapping Analyses Complete!")

if __name__ == "__main__":
    main()
