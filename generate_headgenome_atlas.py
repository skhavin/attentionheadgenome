"""
generate_headgenome_atlas.py
============================
Creates the "HeadGenome Atlas" (Figure 6).
This visualization abandons abstract statistics and simply renders the
literal anatomical matrix (layers x heads) of all 4 architectures side-by-side.

It demonstrates the structural emergence of cognitive circuits
by coloring every single attention head according to the taxonomy.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# ── 1. Load Canonical Data ───────────────────────────────────────────────────
data   = json.load(open("outputs/canonical_labels.json"))
models = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]

# ── 2. Config & Stats ────────────────────────────────────────────────────────
CLASSES = ["Sink", "Local", "Early Induction", "Retrieval", "Late Induction"]

COLORS = {
    "Sink":           "#6c1f8a",   # deep violet
    "Local":          "#1a7a37",   # forest green
    "Retrieval":      "#0d3fbd",   # strong navy blue
    "Early Induction":"#c26b00",   # burnt orange
    "Late Induction": "#b81414",   # deep crimson
}

# ── 3. Figure Layout ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 10), facecolor="#f8f9fa", gridspec_kw={'wspace': 0.15})

total_heads_plotted = 0

for i, model in enumerate(models):
    ax = axes[i]
    ax.set_facecolor("#ffffff")
    
    m_data   = data["models"][model]
    n_layers = m_data["n_layers"]
    heads    = m_data["heads"]
    n_heads  = len(heads) // n_layers
    total_heads_plotted += len(heads)
    
    ax.set_title(f"{model}\n({n_layers} layers $\\times$ {n_heads} heads)", fontsize=13, fontweight="bold")
    
    # Grid parameters
    # Y goes from 0 at top to n_layers-1 at bottom
    # X goes from 0 to n_heads-1
    ax.set_ylim(n_layers, -1)
    ax.set_xlim(-1, n_heads)
    
    # Draw horizontal guide lines for layers
    for layer in range(n_layers):
        ax.axhline(layer, color="#eeeeee", lw=0.8, zorder=0)
    
    # Hide spines
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks(range(n_layers))
    
    if i == 0:
        ax.set_yticklabels([f"L{layer}" for layer in range(n_layers)], fontsize=9, color="#7f8c8d", fontweight="bold")
        ax.set_ylabel("Network Depth (Layers)", fontsize=12, fontweight="bold")
    else:
        ax.set_yticklabels([])
    
    # Plot heads
    for hid, info in heads.items():
        layer = info["layer"]
        head_idx = info["head_idx"]
        depth = info["relative_depth"]
        
        # Resolve label
        label = info["label"]
        if label == "induction":
            label = "Late Induction" if depth >= 0.5 else "Early Induction"
        elif label == "retrieval":
            label = "Retrieval"
        elif label == "local":
            label = "Local"
        elif label == "sink":
            label = "Sink"
            
        color = COLORS[label]
        
        # Scaling marker size based on grid density
        # Llama has 32 heads so we make markers slightly smaller
        base_size = 80 if n_heads <= 16 else 45
        
        if label == "Sink":
            ax.scatter(head_idx, layer, facecolor="none", edgecolor=color, s=base_size, linewidth=1.5, zorder=10)
        elif label == "Local":
            ax.scatter(head_idx, layer, color=color, s=base_size*0.7, alpha=0.35, edgecolors="none", zorder=5)
        else:
            ax.scatter(head_idx, layer, color=color, s=base_size*1.2, alpha=0.9, edgecolors="white", linewidth=0.5, zorder=10)

# ── 4. Legend & Title ─────────────────────────────────────────────────────────
legend_handles = []
# Pre-calculate counts across all models
total_counts = {c: 0 for c in CLASSES}
for model in models:
    for info in data["models"][model]["heads"].values():
        label = info["label"]
        depth = info["relative_depth"]
        if label == "induction": label = "Late Induction" if depth >= 0.5 else "Early Induction"
        elif label == "retrieval": label = "Retrieval"
        elif label == "local": label = "Local"
        elif label == "sink": label = "Sink"
        total_counts[label] += 1

for label in CLASSES:
    n = total_counts[label]
    if label == "Sink":
        h = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                          markeredgecolor=COLORS[label], markersize=12, markeredgewidth=2, label=f"{label} (n={n})")
    else:
        h = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=COLORS[label],
                          markersize=12, label=f"{label} (n={n})")
    legend_handles.append(h)

fig.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02),
           title=f"Head Taxonomy [N={total_heads_plotted}]", fontsize=11, title_fontsize=12,
           framealpha=0.92, edgecolor="#cccccc")

plt.suptitle(
    "HeadGenome Atlas: Anatomical Specialization Across Architectures",
    fontsize=20, fontweight="bold", y=0.96
)
plt.figtext(
    0.5, 0.91,
    "Literal 2D layer-by-head physical matrices tracing the progressive differentiation from Local precursors into specialized cognitive routing circuits.",
    ha="center", fontsize=13, color="#444444"
)

# Adjust layout to make room for legend at bottom and title at top
plt.subplots_adjust(top=0.86, bottom=0.15)

os.makedirs("outputs/final_artifacts", exist_ok=True)
out_path = "outputs/final_artifacts/headgenome_atlas.png"
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")
