import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load V/Q Ratios for GPT-2
with open("../outputs/phase3/weight_features.json", "r") as f:
    weight_data = json.load(f)["models"]["GPT-2"]

vq_ratios = {}
for head_id, info in weight_data.items():
    vq_ratios[head_id] = info["features"]["v_q_ratio"]

# 2. Load True Empirical Labels for GPT-2
with open("../outputs/phase1/gpt2_mechanistic_labels.json", "r") as f:
    labels_data = json.load(f)["heads"]

# 3. Load Entropy Collapse Score (delta) for GPT-2
with open("../outputs/phase1/robust_entropy_gpt2.json", "r") as f:
    entropy_data = json.load(f)["heads"]

rows = []
for head_id, label in labels_data.items():
    if head_id in vq_ratios and head_id in entropy_data:
        delta = entropy_data[head_id]["delta"]
        if delta is not None:
            rows.append({
                "head_id": head_id,
                "label": label,
                "v_q_ratio": vq_ratios[head_id],
                "entropy_collapse_score": delta
            })

df = pd.DataFrame(rows)

species_order = ["sink", "local", "retrieval", "induction"]
colors = {"sink": "#3498db", "local": "#2ecc71", "retrieval": "#f39c12", "induction": "#e74c3c"}

plt.figure(figsize=(12, 8))

for label in species_order:
    subset = df[df["label"] == label]
    plt.scatter(
        subset["v_q_ratio"], 
        subset["entropy_collapse_score"], 
        label=label, 
        color=colors[label], 
        alpha=0.4, 
        s=60
    )

centroids = df.groupby("label")[["v_q_ratio", "entropy_collapse_score"]].mean().reindex(species_order)
centroid_colors = ["#2980b9", "#27ae60", "#d35400", "#c0392b"]

plt.scatter(
    centroids["v_q_ratio"], 
    centroids["entropy_collapse_score"], 
    color=centroid_colors,
    marker='*', 
    s=800, 
    edgecolors='black', 
    zorder=5,
    label="Species Centroids"
)

plt.plot(
    centroids["v_q_ratio"], 
    centroids["entropy_collapse_score"], 
    color="black", 
    linewidth=3, 
    linestyle="--",
    zorder=4,
    label="Developmental Track"
)

for i, row in centroids.iterrows():
    plt.annotate(
        i.capitalize(), 
        (row["v_q_ratio"], row["entropy_collapse_score"]),
        xytext=(10, 10), 
        textcoords="offset points",
        fontsize=14,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.title("Second Developmental Axis: V/Q Ratio vs Entropy Collapse Score", fontsize=18, pad=20)
plt.xlabel("V / Q Matrix Norm Ratio (Developmental Clock)", fontsize=14)
plt.ylabel("Entropy Collapse Score (Delta: Nonmatch - Match)", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig("second_axis_curve.png", dpi=300)
try:
    plt.savefig(r"C:\Users\KHAVIN S\.gemini\antigravity\brain\db51ce35-8b8b-4bcf-90d8-5b2648522b10\second_axis_curve.png", dpi=300)
except Exception as e:
    print("Could not save to artifact directory:", e)

print("\n--- True Centroids ---")
print(centroids)
