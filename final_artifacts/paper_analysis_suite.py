import json
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.utils import resample
from scipy.stats import kruskal, pearsonr, spearmanr, linregress
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

out_dir = "outputs/phase8_paper_suite"
os.makedirs(out_dir, exist_ok=True)

print("Loading data...")
with open("outputs/phase3/weight_features.json", "r") as f:
    data = json.load(f)

rows = []
for model_name, heads in data["models"].items():
    for head_id, info in heads.items():
        layer, head = map(int, head_id.split('_'))
        row = {
            "model": model_name,
            "layer": layer,
            "head": head,
            "label": info["label"],
            **info["features"]
        }
        rows.append(row)

df = pd.DataFrame(rows)
model_layers = df.groupby("model")["layer"].max() + 1
df["relative_depth"] = df.apply(lambda row: row["layer"] / model_layers[row["model"]], axis=1)

results = {}

# --- Helper: Define Early vs Late Induction ---
# We use KMeans(k=2) on v_q_ratio and relative_depth for all induction heads
ind_df = df[df["label"] == "induction"].copy()
scaler = StandardScaler()
X_ind = scaler.fit_transform(ind_df[["relative_depth", "v_q_ratio", "entropy", "diag_off_ratio"]])
kmeans_ind = KMeans(n_clusters=2, random_state=42)
ind_df["subtype"] = kmeans_ind.fit_predict(X_ind)
# Map subtype 0 and 1 to "early" and "late" based on depth
mean_depths = ind_df.groupby("subtype")["relative_depth"].mean()
early_id = mean_depths.idxmin()
ind_df["subtype_name"] = ind_df["subtype"].apply(lambda x: "early" if x == early_id else "late")

print("\n--- 1. Depth-only null control ---")
# Train classifiers: A (depth), B (v/q), C (all), D (shuffled) -> early/late
X_depth = ind_df[["relative_depth"]]
X_vq = ind_df[["v_q_ratio"]]
X_all = ind_df[[c for c in ind_df.columns if c.startswith("svd") or c in ["entropy", "diag_off_ratio", "v_q_ratio", "q_k_ratio", "relative_depth"]]]
y = (ind_df["subtype_name"] == "late").astype(int)

clf = LogisticRegression()
acc_depth = cross_val_score(clf, X_depth, y, cv=5).mean()
acc_vq = cross_val_score(clf, X_vq, y, cv=5).mean()
acc_all = cross_val_score(clf, StandardScaler().fit_transform(X_all), y, cv=5).mean()

y_shuffled = np.random.permutation(y)
acc_shuffled = cross_val_score(clf, X_depth, y_shuffled, cv=5).mean()

results["null_control"] = {
    "acc_depth_only": acc_depth,
    "acc_vq_only": acc_vq,
    "acc_all_features": acc_all,
    "acc_shuffled": acc_shuffled
}
print(f"Depth only: {acc_depth:.3f}, V/Q only: {acc_vq:.3f}, All features: {acc_all:.3f}, Shuffled: {acc_shuffled:.3f}")

print("\n--- 2. Per-model replication ---")
per_model_repl = {}
for model in df["model"].unique():
    m_ind = ind_df[ind_df["model"] == model]
    if len(m_ind) < 5:
        continue
    m_X = scaler.fit_transform(m_ind[["relative_depth", "v_q_ratio"]])
    km = KMeans(n_clusters=2, random_state=42).fit(m_X)
    centers = km.cluster_centers_
    # Original unscaled centers
    m_X_unscaled = m_ind[["relative_depth", "v_q_ratio"]]
    m_ind["cluster"] = km.labels_
    c_means = m_ind.groupby("cluster")[["relative_depth", "v_q_ratio"]].mean()
    
    depth_diff = abs(c_means.iloc[0]["relative_depth"] - c_means.iloc[1]["relative_depth"])
    vq_diff = abs(c_means.iloc[0]["v_q_ratio"] - c_means.iloc[1]["v_q_ratio"])
    
    per_model_repl[model] = {
        "early_depth": float(c_means["relative_depth"].min()),
        "late_depth": float(c_means["relative_depth"].max()),
        "depth_separation": float(depth_diff),
        "vq_separation": float(vq_diff),
        "exists": bool(depth_diff > 0.15 and vq_diff > 0.15)
    }
    print(f"{model}: early/late split exists? {per_model_repl[model]['exists']} (depth diff={depth_diff:.2f}, vq diff={vq_diff:.2f})")
results["per_model_replication"] = per_model_repl

print("\n--- 3. Bootstrap stability ---")
n_bootstraps = 100
boot_results = []
original_labels = ind_df["subtype"].values

for i in range(n_bootstraps):
    # Resample indices
    indices = resample(np.arange(len(ind_df)), random_state=i)
    X_boot = X_ind[indices]
    km = KMeans(n_clusters=2, random_state=i).fit(X_boot)
    
    # Map back to full dataset to get ARI
    full_pred = km.predict(X_ind)
    ari = adjusted_rand_score(original_labels, full_pred)
    
    # Centers in original space
    boot_df = ind_df.iloc[indices].copy()
    boot_df["boot_label"] = km.labels_
    c_means = boot_df.groupby("boot_label")[["relative_depth", "v_q_ratio"]].mean()
    
    e_depth = c_means["relative_depth"].min()
    l_depth = c_means["relative_depth"].max()
    
    boot_results.append({
        "ari": ari,
        "early_depth": e_depth,
        "late_depth": l_depth
    })

boot_df_res = pd.DataFrame(boot_results)
ari_mean, ari_std = boot_df_res["ari"].mean(), boot_df_res["ari"].std()
print(f"Bootstrap ARI: {ari_mean:.3f} +- {ari_std:.3f}")
results["bootstrap_stability"] = {
    "ari_mean": float(ari_mean),
    "ari_std": float(ari_std),
    "early_depth_mean": float(boot_df_res["early_depth"].mean()),
    "early_depth_std": float(boot_df_res["early_depth"].std()),
    "late_depth_mean": float(boot_df_res["late_depth"].mean()),
    "late_depth_std": float(boot_df_res["late_depth"].std())
}

print("\n--- Checklist B. Spatial laws ---")
depths_by_species = [df[df["label"] == sp]["relative_depth"].values for sp in ["sink", "local", "retrieval", "induction"]]
kruskal_stat, kruskal_p = kruskal(*depths_by_species)
print(f"Kruskal-Wallis across species depth: H={kruskal_stat:.2f}, p={kruskal_p:.2e}")
results["spatial_laws"] = {
    "kruskal_h": float(kruskal_stat),
    "kruskal_p": float(kruskal_p),
    "means": df.groupby("label")["relative_depth"].mean().to_dict()
}

print("\n--- Checklist C. V/Q developmental law ---")
dev_laws = {}
for model in df["model"].unique():
    m_df = df[df["model"] == model]
    pearson_r, p_p = pearsonr(m_df["relative_depth"], m_df["v_q_ratio"])
    spearman_rho, s_p = spearmanr(m_df["relative_depth"], m_df["v_q_ratio"])
    dev_laws[model] = {
        "pearson": float(pearson_r),
        "spearman": float(spearman_rho)
    }
    print(f"{model}: Pearson={pearson_r:.3f}, Spearman={spearman_rho:.3f}")

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(df["relative_depth"], df["v_q_ratio"])
print(f"Global OLS slope: {slope:.3f} (p={p_value:.2e})")
dev_laws["global_ols_slope"] = float(slope)
dev_laws["global_ols_pvalue"] = float(p_value)
results["vq_developmental_law"] = dev_laws

print("\nSaving results...")
with open(os.path.join(out_dir, "statistical_suite_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Done. Saved to outputs/phase8_paper_suite/")

# Generate causal patching scaffold script
scaffold = """
# This is a scaffolding script for running the causal patching tests (Items 4-10).
# Requires transformer_lens or similar mechanistic interpretability library.

def run_causal_patching_induction_task(model, early_heads, late_heads):
    \"\"\"
    Task 4 & 6 & 7: Matching vs Copying Causal Test
    Prompt: 'A B C ... A B'
    Expected: 'C'
    \"\"\"
    # 1. Clean run
    clean_logits = model.run(prompt)
    
    # 2. Corrupted run (e.g. 'A B X ... A B' where expected is X)
    corrupted_cache = model.run_with_cache(corrupted_prompt)
    
    # 3. Patching interventions
    # a. Patch early Q/K -> expect failure to locate (attention mass shifts)
    # b. Patch late V -> expect failure to copy (attends correctly, outputs wrong token)
    
    pass

def run_attention_target_analysis(model, heads, prompt):
    \"\"\"
    Task 5: Attention Target Analysis
    \"\"\"
    _, cache = model.run_with_cache(prompt)
    # Extract attention patterns for `heads`
    # Compute mass on previous prefix, copied token, local window
    pass
"""
with open(os.path.join(out_dir, "causal_patching_scaffold.py"), "w") as f:
    f.write(scaffold)
