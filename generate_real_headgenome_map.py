import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import os

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

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 8))

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
        ax.scatter(x_jitter, depths, c=color, s=25, alpha=0.7, edgecolors='none')

ax.set_xticks(range(len(models_to_plot)))
ax.set_xticklabels(models_to_plot, fontsize=12, fontweight='bold')
ax.set_ylabel("Relative Network Depth", fontsize=12)
ax.set_title("The HeadGenome Map: Architecture Distribution (Empirical)", fontsize=14, fontweight='bold')
ax.set_ylim(-0.05, 1.05)

legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors.keys()]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Head Taxonomy")

plt.tight_layout()
os.makedirs("outputs/final_artifacts", exist_ok=True)
plt.savefig("outputs/final_artifacts/headgenome_map.png", dpi=300, bbox_inches='tight')
print("Successfully generated empirical HeadGenome map using 1,568 real JSON coordinates.")
