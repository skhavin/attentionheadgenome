import json
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import os
import scipy.stats as stats

with open("outputs/phase3/weight_features.json", "r") as f:
    data = json.load(f)

rows = []
for model_name, heads in data["models"].items():
    if model_name not in ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]:
        continue
        
    for head_id, info in heads.items():
        layer, head = map(int, head_id.split('_'))
        label = info["label"]
        rows.append({
            "model": model_name,
            "layer": layer,
            "head": head,
            "label": label
        })

df = pd.DataFrame(rows)
model_layers = df.groupby("model")["layer"].max() + 1
df["relative_depth"] = df.apply(lambda row: row["layer"] / model_layers[row["model"]], axis=1)

# Split Induction into Early and Late based on depth
def refine_label(row):
    if row["label"] == "induction":
        if row["relative_depth"] < 0.5:
            return "Early Induction"
        else:
            return "Late Induction"
    return row["label"].capitalize()

df["refined_label"] = df.apply(refine_label, axis=1)

# Statistical Analysis
print("--- Statistical Distribution of Relative Depth ---")
classes = ["Sink", "Local", "Retrieval", "Early Induction", "Late Induction"]
groups = []
for cls in classes:
    depths = df[df["refined_label"] == cls]["relative_depth"].values
    groups.append(depths)
    print(f"{cls:<15}: {np.mean(depths):.2f} ± {np.std(depths):.2f}")

h_stat, p_val = stats.kruskal(*groups)
print(f"\nKruskal-Wallis Test: H={h_stat:.2f}, p={p_val:.2e}")

# Set up the plot (2 panels: Scatter + KDE)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})

models_to_plot = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]
colors = {"Sink": "purple", "Local": "green", "Retrieval": "blue", "Early Induction": "orange", "Late Induction": "red"}

np.random.seed(42)
for i, model in enumerate(models_to_plot):
    subset = df[df["model"] == model]
    
    for label, color in colors.items():
        sub_subset = subset[subset["refined_label"] == label]
        if len(sub_subset) == 0:
            continue
            
        depths = sub_subset["relative_depth"].values
        # Add slight jitter to x-axis to prevent total overlap
        x_jitter = np.random.normal(i, 0.08, len(depths))
        ax1.scatter(x_jitter, depths, c=color, s=25, alpha=0.7, edgecolors='none')

ax1.set_xticks(range(len(models_to_plot)))
ax1.set_xticklabels(models_to_plot, fontsize=12, fontweight='bold')
ax1.set_ylabel("Relative Network Depth", fontsize=12, fontweight='bold')
ax1.set_title("Spatial Distribution Across Architectures", fontsize=14, fontweight='bold')
ax1.set_ylim(-0.05, 1.05)

legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors.keys()]
ax1.legend(handles=legend_elements, loc='upper left', title="Head Taxonomy")

# Panel 2: KDE Density
from scipy.stats import gaussian_kde
y_grid = np.linspace(-0.05, 1.05, 200)

for label, color in colors.items():
    subset = df[df["refined_label"] == label]
    if len(subset) > 1:
        kde = gaussian_kde(subset["relative_depth"])
        density = kde(y_grid)
        ax2.plot(density, y_grid, color=color, label=label)
        ax2.fill_betweenx(y_grid, 0, density, color=color, alpha=0.3)

ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel("Density", fontsize=12, fontweight='bold')
ax2.set_title("Global Density", fontsize=14, fontweight='bold')
ax2.set_yticks([]) # Hide y-ticks on the second axis

plt.suptitle("Spatial Distribution of Functional Attention Head Types Across Transformer Architectures", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

os.makedirs("outputs/final_artifacts", exist_ok=True)
plt.savefig("outputs/final_artifacts/headgenome_map.png", dpi=300, bbox_inches='tight')
print("Successfully generated empirical HeadGenome map with Density Plots.")
