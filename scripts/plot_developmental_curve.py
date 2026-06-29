import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
            "v_q_ratio": info["features"]["v_q_ratio"]
        }
        rows.append(row)

df = pd.DataFrame(rows)
model_layers = df.groupby("model")["layer"].max() + 1
df["relative_depth"] = df.apply(lambda row: row["layer"] / model_layers[row["model"]], axis=1)

# Order of species as hypothesized by user
species_order = ["sink", "local", "retrieval", "induction"]
colors = {"sink": "#3498db", "local": "#2ecc71", "retrieval": "#f39c12", "induction": "#e74c3c"}

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot scatter
for label in species_order:
    subset = df[df["label"] == label]
    plt.scatter(
        subset["relative_depth"], 
        subset["v_q_ratio"], 
        label=label, 
        color=colors[label], 
        alpha=0.4, 
        s=60
    )

# Calculate and plot centroids
centroids = df.groupby("label")[["relative_depth", "v_q_ratio"]].mean().reindex(species_order)
centroid_colors = ["#2980b9", "#27ae60", "#d35400", "#c0392b"]

plt.scatter(
    centroids["relative_depth"], 
    centroids["v_q_ratio"], 
    color=centroid_colors,
    marker='*', 
    s=800, 
    edgecolors='black', 
    zorder=5,
    label="Species Centroids"
)

# Connect centroids with a line to show trajectory
plt.plot(
    centroids["relative_depth"], 
    centroids["v_q_ratio"], 
    color="black", 
    linewidth=3, 
    linestyle="--",
    zorder=4,
    label="Developmental Track"
)

# Annotate centroids
for i, row in centroids.iterrows():
    plt.annotate(
        i.capitalize(), 
        (row["relative_depth"], row["v_q_ratio"]),
        xytext=(10, 10), 
        textcoords="offset points",
        fontsize=14,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.title("Attention Heads: Discrete Species vs Continuous Developmental Process", fontsize=18, pad=20)
plt.xlabel("Relative Depth (Layer / Total Layers)", fontsize=14)
plt.ylabel("V / Q Matrix Norm Ratio", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save locally and to artifact dir
plt.savefig("outputs/developmental_curve.png", dpi=300)
try:
    plt.savefig(r"C:\Users\KHAVIN S\.gemini\antigravity\brain\db51ce35-8b8b-4bcf-90d8-5b2648522b10\developmental_curve.png", dpi=300)
except Exception as e:
    print("Could not save to artifact directory:", e)

print("\n--- Centroids ---")
print(centroids)

# Check standard deviation along the curve
df_sorted = df.sort_values(by="relative_depth")
print("\nCorrelation (Depth vs V/Q Ratio):", df["relative_depth"].corr(df["v_q_ratio"]))
