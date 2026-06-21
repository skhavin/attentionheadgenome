import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

def main():
    with open("outputs/phase7/head_audit.json") as f:
        data = json.load(f)
        
    print("Total entries in audit:", len(data))
    
    heads = {}
    for x in data:
        k = (x["layer"], x["head"])
        if k not in heads:
            heads[k] = []
        heads[k].append(x)
        
    errors = []
    head_keys = []
    
    for k, entries in heads.items():
        best_entry = min(entries, key=lambda e: e["out_l_inf_natural_max"] if e["out_l_inf_natural_max"] is not None else 999)
        
        e_nat = best_entry["out_l_inf_natural_max"]
        e_copy = best_entry["out_l_inf_copy_max"]
        
        if e_nat is None or e_copy is None:
            continue
            
        errors.append([np.log10(e_nat + 1e-10), np.log10(e_copy + 1e-10)])
        head_keys.append((k, best_entry["type"]))
        
    errors = np.array(errors)
    print("Shape of errors:", errors.shape)
    
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(errors)
    labels = gmm.predict(errors)
    
    centroids = gmm.means_
    
    # Sort centroids
    # Tier 3: highest nat
    # Tier 1: out of remaining two, lowest copy
    # Tier 2: the other one
    
    tier_map = {}
    sorted_by_nat = np.argsort(centroids[:, 0])
    t3_idx = sorted_by_nat[-1]
    
    rem = [i for i in range(3) if i != t3_idx]
    if centroids[rem[0], 1] < centroids[rem[1], 1]:
        t1_idx, t2_idx = rem[0], rem[1]
    else:
        t1_idx, t2_idx = rem[1], rem[0]
        
    tier_names = {t1_idx: 'Tier 1 (Safe)', t2_idx: 'Tier 2 (Regime-Switch)', t3_idx: 'Tier 3 (Full Attn)'}
    
    plt.figure(figsize=(10, 8))
    colors = {t1_idx: 'green', t2_idx: 'orange', t3_idx: 'red'}
    
    for i in range(3):
        mask = labels == i
        plt.scatter(errors[mask, 0], errors[mask, 1], c=colors[i], label=f'{tier_names[i]} (n={mask.sum()})', alpha=0.6)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    # Plot x=y line for reference
    min_val = min(errors[:, 0].min(), errors[:, 1].min())
    max_val = max(errors[:, 0].max(), errors[:, 1].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='x=y')
    
    plt.xlabel('log10(L_inf absolute output error - Natural)')
    plt.ylabel('log10(L_inf absolute output error - Copy)')
    plt.title('GMM Clustering of Attention Heads (3 components)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('outputs/phase7/plots', exist_ok=True)
    out_path = 'outputs/phase7/plots/gmm_clusters.png'
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    
    # Print counts for each tier
    t1_count = (labels == t1_idx).sum()
    t2_count = (labels == t2_idx).sum()
    t3_count = (labels == t3_idx).sum()
    print(f"Tier 1: {t1_count}")
    print(f"Tier 2: {t2_count}")
    print(f"Tier 3: {t3_count}")

    # Also write a JSON file with tier assignments
    tier_assignments = []
    for (k, htype), l in zip(head_keys, labels):
        tier = 1 if l == t1_idx else (2 if l == t2_idx else 3)
        tier_assignments.append({
            "layer": k[0],
            "head": k[1],
            "type": htype,
            "tier": tier
        })
        
    with open("outputs/phase7/head_tiers_gmm.json", "w") as f:
        json.dump(tier_assignments, f, indent=2)
    print("Saved tier assignments to outputs/phase7/head_tiers_gmm.json")

if __name__ == "__main__":
    main()
