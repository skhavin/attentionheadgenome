import json
import numpy as np
import pandas as pd

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

print("=== 1. V/Q Ratio vs Depth Scaling Law ===")
for model in df["model"].unique():
    subset = df[df["model"] == model]
    corr = subset["relative_depth"].corr(subset["v_q_ratio"])
    print(f"{model}: v_q_ratio correlation with depth = {corr:.3f}")

print("\n=== 2. Architectural Scaling: GQA vs Retrieval ===")
counts = df.groupby(["model", "label"]).size().unstack(fill_value=0)
counts["total"] = counts.sum(axis=1)
for label in ["induction", "local", "retrieval", "sink"]:
    counts[f"{label}_pct"] = counts[label] / counts["total"] * 100

print(counts[["total", "retrieval", "retrieval_pct", "induction_pct", "sink_pct", "local_pct"]])

print("\n=== 3. Induction Sub-Types Analysis ===")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ind_df = df[df["label"] == "induction"].copy()
features = ["entropy", "diag_off_ratio", "v_q_ratio", "q_k_ratio"]
X = ind_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
ind_df["sub_type"] = kmeans.fit_predict(X_scaled)

for sub in range(3):
    sub_df = ind_df[ind_df["sub_type"] == sub]
    print(f"\nSub-Type {sub} (n={len(sub_df)}):")
    print(f"  Mean Depth: {sub_df['relative_depth'].mean():.3f}")
    for feat in features:
        print(f"  {feat}: {sub_df[feat].mean():.3f}")

print("\n=== 4. Local Sub-Types Analysis ===")
loc_df = df[df["label"] == "local"].copy()
X_loc = loc_df[features]
X_loc_scaled = scaler.fit_transform(X_loc)

kmeans_loc = KMeans(n_clusters=2, random_state=42, n_init=10)
loc_df["sub_type"] = kmeans_loc.fit_predict(X_loc_scaled)

for sub in range(2):
    sub_df = loc_df[loc_df["sub_type"] == sub]
    print(f"\nLocal Sub-Type {sub} (n={len(sub_df)}):")
    print(f"  Mean Depth: {sub_df['relative_depth'].mean():.3f}")
    for feat in features:
        print(f"  {feat}: {sub_df[feat].mean():.3f}")

