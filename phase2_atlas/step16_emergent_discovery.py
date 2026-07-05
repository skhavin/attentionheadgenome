"""
step16_emergent_discovery.py  (Workstream 1 - Unsupervised Discovery)
----------------------------------------------------------------------
Builds a flat feature matrix from ALL heads across ALL 4 models,
combining existing atlas data + new step15 rich features, then runs:

  1. PCA  - find major axes of variation
  2. UMAP + HDBSCAN clustering  - find emergent groups
  3. Correlation heatmap  - find unexpected pairwise associations
  4. Per-cluster characterization  - report what each emergent cluster means

Does NOT assume the 4 pre-labeled classes (Local, Sink, Induction, Retrieval)
— the goal is to discover what the data says without pre-specification.
"""
import json, os, numpy as np
from collections import defaultdict, Counter

MODELS = ["gpt2-medium", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Llama-3.2-1B"]

os.makedirs("outputs/routing", exist_ok=True)

# ── 1. Build feature matrix ───────────────────────────────────────────────────
print("Building feature matrix...")
rows   = []   # one per head
labels = []   # pre-existing class label (kept for post-hoc comparison only)
head_ids = [] # (model, layer, head) for traceability

FEATURES = [
    "entropy_delta", "bos_mass", "local_mass", "long_range_mass",
    "mean_distance", "vq_ratio", "mean_output_norm",
    "nsubj_mass", "obj_mass", "punct_mass",
    "mean_max_attn", "mean_entropy",
    "entropy_no_bos_delta",
    "pos_bias_early", "pos_bias_middle", "pos_bias_late",
    "activation_sparsity", "inter_layer_corr",
]

for model in MODELS:
    atlas_path   = f"outputs/phase2_atlas/{model}_head_atlas.json"
    rich_path    = f"outputs/routing/{model}_rich_features.json"

    if not os.path.exists(atlas_path):
        print(f"  MISSING atlas for {model}, skipping.")
        continue
    if not os.path.exists(rich_path):
        print(f"  MISSING rich features for {model}, skipping.")
        continue

    with open(atlas_path)  as f: atlas = json.load(f)
    with open(rich_path)   as f: rich  = json.load(f)

    for k, head in atlas["heads"].items():
        r = rich["heads"].get(k, {})
        geom = head.get("attention_geometry", {})
        gram = head.get("grammar_profile", {})
        sat  = head.get("softmax_saturation", {})
        sink = head.get("sink_falsification", {})
        ent  = head.get("entropy_profile", {})

        # Skip heads with missing essential data
        if not geom or not sat:
            continue

        pos_bias = r.get("position_bias", {})

        row = [
            ent.get("delta_collapse", 0.0) or 0.0,
            geom.get("bos_mass", 0.0),
            geom.get("local_mass", 0.0),
            geom.get("long_range_mass", 0.0),
            geom.get("mean_distance", 0.0),
            head.get("vq_ratio", 0.0) or 0.0,
            head.get("mean_output_norm", 0.0) or 0.0,
            gram.get("nsubj", 0.0),
            gram.get("obj", 0.0),
            gram.get("punct", 0.0),
            sat.get("mean_max_attn", 0.0),
            sat.get("mean_entropy", 0.0),
            sink.get("delta_no_bos", 0.0) or 0.0,
            pos_bias.get("early_third", 0.0),
            pos_bias.get("middle_third", 0.0),
            pos_bias.get("late_third", 0.0),
            r.get("activation_sparsity", 0.0) or 0.0,
            r.get("inter_layer_corr", 0.0) or 0.0,
        ]

        rows.append(row)
        labels.append(head.get("class_label", "Unknown"))
        head_ids.append((model, head["layer"], head["head"]))

X = np.array(rows, dtype=float)
print(f"Feature matrix: {X.shape[0]} heads x {X.shape[1]} features")

# Handle NaN
nan_mask = np.isnan(X)
col_means = np.nanmean(X, axis=0)
for j in range(X.shape[1]):
    X[nan_mask[:, j], j] = col_means[j]

# Standardise
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# ── 2. PCA ────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print("\n=== PCA Explained Variance ===")
cumvar = np.cumsum(pca.explained_variance_ratio_)
for i, (v, cv) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
    print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {cv*100:.1f}%)")

print("\nTop feature loadings for PC1:")
pc1 = pca.components_[0]
top_idx = np.argsort(np.abs(pc1))[::-1][:5]
for i in top_idx:
    print(f"  {FEATURES[i]}: {pc1[i]:+.3f}")

print("\nTop feature loadings for PC2:")
pc2 = pca.components_[1]
top_idx = np.argsort(np.abs(pc2))[::-1][:5]
for i in top_idx:
    print(f"  {FEATURES[i]}: {pc2[i]:+.3f}")

# ── 3. UMAP + HDBSCAN ────────────────────────────────────────────────────────
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("\nWARNING: umap-learn not installed. Falling back to PCA-only clustering.")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

if HAS_UMAP:
    print("\nRunning UMAP (2D projection)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
else:
    X_umap = X_pca[:, :2]

if HAS_HDBSCAN:
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
    cluster_labels = clusterer.fit_predict(X_umap)
else:
    # Fallback: KMeans with k=6
    from sklearn.cluster import KMeans
    print("Running KMeans (k=6) as fallback...")
    clusterer = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(X_scaled)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Found {n_clusters} emergent clusters.")

# ── 4. Cluster characterization ───────────────────────────────────────────────
print("\n=== EMERGENT CLUSTER CHARACTERIZATION ===")
cluster_report = {}

for c in sorted(set(cluster_labels)):
    if c == -1:
        print(f"\nCluster -1 (noise): {(cluster_labels==-1).sum()} heads")
        continue

    mask = cluster_labels == c
    X_c  = X[mask]
    labs = [labels[i] for i, m in enumerate(mask) if m]
    mods = [head_ids[i][0] for i, m in enumerate(mask) if m]

    label_dist = Counter(labs)
    model_dist = Counter(mods)

    # Feature centroids
    centroids = {FEATURES[j]: round(float(X_c[:, j].mean()), 4) for j in range(len(FEATURES))}

    # Top distinguishing features (high absolute mean, normalized by global std)
    global_std = X_scaled.std(axis=0) + 1e-8
    X_c_scaled = (X_c - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    feature_devs = X_c_scaled.mean(axis=0)
    top_features = sorted(zip(FEATURES, feature_devs), key=lambda x: abs(x[1]), reverse=True)[:5]

    print(f"\nCluster {c} (N={mask.sum()}):")
    print(f"  Pre-labeled classes: {dict(label_dist)}")
    print(f"  Models:              {dict(model_dist)}")
    print(f"  Top distinguishing features:")
    for feat, dev in top_features:
        print(f"    {feat}: {dev:+.3f} std devs from mean")

    cluster_report[int(c)] = {
        "n": int(mask.sum()),
        "label_distribution": dict(label_dist),
        "model_distribution": dict(model_dist),
        "feature_centroids": centroids,
        "top_distinguishing": [(f, round(d, 4)) for f, d in top_features],
    }

# ── 5. Correlation heatmap (top unexpected associations) ─────────────────────
print("\n=== TOP PAIRWISE FEATURE CORRELATIONS ===")
corr_matrix = np.corrcoef(X_scaled.T)
pairs = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        pairs.append((abs(corr_matrix[i,j]), corr_matrix[i,j], FEATURES[i], FEATURES[j]))
pairs.sort(reverse=True)

print("Top 10 correlated feature pairs (unexpected pairs only):")
shown = 0
expected_pairs = {
    ("bos_mass", "local_mass"), ("bos_mass", "mean_distance"),
    ("local_mass", "mean_distance"), ("vq_ratio", "mean_output_norm"),
}
for abs_r, r, f1, f2 in pairs:
    if (f1, f2) not in expected_pairs and (f2, f1) not in expected_pairs:
        print(f"  {f1} x {f2}: r={r:.3f}")
        shown += 1
        if shown >= 10:
            break

# Save results
out = {
    "n_heads": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    "n_emergent_clusters": n_clusters,
    "clusters": cluster_report,
}
with open("outputs/routing/emergent_discovery_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to outputs/routing/emergent_discovery_results.json")
