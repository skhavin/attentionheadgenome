"""
generate_headgenome_atlas.py
============================
Creates the 3-panel iconic HeadGenome Atlas.

Panel A: Scatter plot (the evidence)
Panel B: Density plot + Trajectory Means (statistical summary)
Panel C: The "Transformer Anatomy" showing GPT-2 Medium's exact 24x16 
         head grid, vividly illustrating specialization emerging across depth.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, kruskal
import os

# ── 1. Load Canonical Data ───────────────────────────────────────────────────
data   = json.load(open("outputs/canonical_labels.json"))
models = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]

rows = []
for model in models:
    m_data   = data["models"][model]
    n_layers = m_data["n_layers"]
    for hid, info in m_data["heads"].items():
        label = info["label"]
        depth = info["relative_depth"]
        if label == "induction":
            label = "Late Induction" if depth >= 0.5 else "Early Induction"
        elif label == "retrieval":
            label = "Retrieval"
        elif label == "local":
            label = "Local"
        elif label == "sink":
            label = "Sink"
        rows.append({"model": model, "depth": depth, "label": label, "layer": info["layer"], "head_idx": info["head_idx"]})

# ── 2. Config & Stats ────────────────────────────────────────────────────────
CLASSES = ["Sink", "Local", "Early Induction", "Retrieval", "Late Induction"]
RENDER_ORDER = ["Local", "Early Induction", "Sink", "Late Induction", "Retrieval"]

COLORS = {
    "Sink":           "#6c1f8a",   # deep violet
    "Local":          "#1a7a37",   # forest green
    "Retrieval":      "#0d3fbd",   # strong navy blue
    "Early Induction":"#c26b00",   # burnt orange
    "Late Induction": "#b81414",   # deep crimson
}

SCATTER_CFG = {
    "Local":           {"s": 15,  "alpha": 0.25, "zorder": 1, "edgecolors": "none"},
    "Early Induction": {"s": 45,  "alpha": 0.85, "zorder": 4, "edgecolors": "none"},
    "Late Induction":  {"s": 50,  "alpha": 0.85, "zorder": 5, "edgecolors": "none"},
    "Retrieval":       {"s": 90,  "alpha": 1.00, "zorder": 7, "edgecolors": "white", "linewidths": 0.8},
    "Sink":            {"s": 80,  "alpha": 1.00, "zorder": 6, "facecolors": "none",
                        "edgecolors": "#6c1f8a", "linewidths": 2.2},
}

depths_by_class = {c: [] for c in CLASSES}
for r in rows:
    depths_by_class[r["label"]].append(r["depth"])

n_by_class = {c: len(v) for c, v in depths_by_class.items()}
total_n = sum(n_by_class.values())

groups = [np.array(depths_by_class[c]) for c in CLASSES]
H, p = kruskal(*groups)
k = len(CLASSES)
eta_sq = (H - k + 1) / (total_n - k)

# ── 3. Figure Layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9), facecolor="#f8f9fa")
gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.2, 1.7], wspace=0.15)

ax1 = fig.add_subplot(gs[0])  # Scatter
ax2 = fig.add_subplot(gs[1])  # Density
ax3 = fig.add_subplot(gs[2])  # Transformer Anatomy

ax1.set_facecolor("#ffffff")
ax2.set_facecolor("#ffffff")
ax3.set_facecolor("#ffffff")

# ── 4. Panel A: Scatter ──────────────────────────────────────────────────────
np.random.seed(42)
for label in RENDER_ORDER:
    cfg = SCATTER_CFG[label]
    color = COLORS[label]

    for i, model in enumerate(models):
        pts = np.array([r["depth"] for r in rows if r["model"] == model and r["label"] == label])
        if len(pts) == 0: continue
        x = np.random.normal(i, 0.08, len(pts))

        if label == "Sink":
            ax1.scatter(x, pts, facecolors=cfg["facecolors"], edgecolors=cfg["edgecolors"],
                        linewidths=cfg["linewidths"], s=cfg["s"], alpha=cfg["alpha"], zorder=cfg["zorder"])
        elif label == "Retrieval":
            ax1.scatter(x, pts, c=color, edgecolors=cfg["edgecolors"], linewidths=cfg["linewidths"],
                        s=cfg["s"], alpha=cfg["alpha"], zorder=cfg["zorder"])
        else:
            ax1.scatter(x, pts, c=color, edgecolors=cfg["edgecolors"], s=cfg["s"],
                        alpha=cfg["alpha"], zorder=cfg["zorder"])

ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=12, fontweight="bold")
ax1.set_ylabel("Relative Network Depth", fontsize=12, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("A. Cross-Architecture Distribution", fontsize=14, fontweight="bold", loc="left")
ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#cccccc")
ax1.spines[["top", "right"]].set_visible(False)

legend_handles = []
for label in CLASSES:
    n = n_by_class[label]
    if label == "Sink":
        h = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                          markeredgecolor=COLORS[label], markersize=10, markeredgewidth=2, label=f"{label} (n={n})")
    else:
        h = mpatches.Patch(facecolor=COLORS[label], label=f"{label} (n={n})")
    legend_handles.append(h)

ax1.legend(handles=legend_handles, loc="upper left", title=f"Head Taxonomy [N={total_n}]",
           fontsize=10, title_fontsize=11, framealpha=0.92, edgecolor="#cccccc")

# ── 5. Panel B: Density + Trajectory ─────────────────────────────────────────
y_grid = np.linspace(-0.05, 1.05, 400)
all_densities = {}

for label in CLASSES:
    pts = np.array(depths_by_class[label])
    if len(pts) < 5:
        all_densities[label] = np.zeros_like(y_grid)
        continue
    kde = gaussian_kde(pts, bw_method="scott")
    all_densities[label] = kde(y_grid) * len(pts)

means = {}
for label in CLASSES:
    pts = np.array(depths_by_class[label])
    if len(pts) >= 5:
        density = all_densities[label] / all_densities[label].max()
        means[label] = np.mean(pts)
        color = COLORS[label]
        lw = 1.5 if label == "Local" else 2.5
        alpha_f = 0.10 if label == "Local" else 0.25
        ax2.plot(density, y_grid, color=color, linewidth=lw)
        ax2.fill_betweenx(y_grid, 0, density, color=color, alpha=alpha_f)

ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(0, 1.35)
ax2.set_xlabel("Norm. Density", fontsize=11, fontweight="bold")
ax2.set_title("B. Global Depth Trajectory", fontsize=14, fontweight="bold", loc="left")
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels([]) # Hide labels
ax2.tick_params(axis="y", left=False)
ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#cccccc")
ax2.spines[["top", "right", "left"]].set_visible(False)

# Overlay the mean trajectory line
traj_order = ["Sink", "Early Induction", "Local", "Retrieval", "Late Induction"]
traj_x = []
traj_y = []
for label in traj_order:
    if label in means:
        y_val = means[label]
        x_val = 1.15
        traj_x.append(x_val)
        traj_y.append(y_val)
        
        # Hollow for sink
        if label == "Sink":
            ax2.scatter(x_val, y_val, facecolor="none", edgecolor=COLORS[label], s=100, zorder=10, linewidth=2.5)
        else:
            ax2.scatter(x_val, y_val, color=COLORS[label], s=100, zorder=10, edgecolor="white", linewidth=1.5)
        
        ax2.text(x_val + 0.05, y_val, f"Mean: {y_val:.2f}", va="center", fontsize=9, fontweight="bold", color=COLORS[label])

ax2.plot(traj_x, traj_y, color="#7f8c8d", linestyle="--", linewidth=2.5, alpha=0.7, zorder=5)

# ── 6. Panel C: Transformer Anatomy (GPT-2 Stack) ────────────────────────────
ax3.axis("off")
ax3.set_title("C. Anatomical Maturation (GPT-2)", fontsize=14, fontweight="bold", loc="left")

gpt2_rows = [r for r in rows if r["model"] == "GPT-2"]
num_layers = 24
num_heads = 16

# Draw Input block
y_offset = 0.95
ax3.text(0.5, y_offset, "Input Sequence", ha="center", va="center", fontsize=11, fontweight="bold", color="black", 
         bbox=dict(facecolor="#ecf0f1", edgecolor="#bdc3c7", boxstyle="round,pad=0.3"))

# Draw downward arrow
ax3.annotate("", xy=(0.5, y_offset - 0.04), xytext=(0.5, y_offset - 0.01),
             arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

# Grid parameters
grid_top = y_offset - 0.07
grid_bottom = 0.1
layer_height = (grid_top - grid_bottom) / (num_layers - 1)

# Draw the exact head matrix
for layer in range(num_layers):
    layer_heads = [r for r in gpt2_rows if r["layer"] == layer]
    
    y_pos = grid_top - layer * layer_height
    
    # Layer label
    ax3.text(0.05, y_pos, f"L{layer}", ha="right", va="center", fontsize=8, color="#7f8c8d", fontweight="bold")
    
    for head_idx in range(num_heads):
        # find the head
        head_data = next((h for h in layer_heads if h["head_idx"] == head_idx), None)
        x_pos = 0.15 + (head_idx / (num_heads - 1)) * 0.75
        
        if head_data:
            label = head_data["label"]
            color = COLORS[label]
            if label == "Sink":
                ax3.scatter(x_pos, y_pos, facecolor="none", edgecolor=color, s=45, linewidth=1.5, zorder=10)
            elif label == "Local":
                ax3.scatter(x_pos, y_pos, color=color, s=30, alpha=0.3, edgecolors="none", zorder=5)
            else:
                ax3.scatter(x_pos, y_pos, color=color, s=50, alpha=0.9, edgecolors="white", linewidth=0.5, zorder=10)

# Draw downward arrow to output
ax3.annotate("", xy=(0.5, grid_bottom - 0.04), xytext=(0.5, grid_bottom - 0.01),
             arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

# Draw Output block
ax3.text(0.5, grid_bottom - 0.07, "Output Representation", ha="center", va="center", fontsize=11, fontweight="bold", color="black",
         bbox=dict(facecolor="#ecf0f1", edgecolor="#bdc3c7", boxstyle="round,pad=0.3"))

# Add connecting flow lines down the sides to emphasize "depth"
ax3.annotate("", xy=(0.95, grid_bottom + 0.05), xytext=(0.95, grid_top - 0.05),
             arrowprops=dict(arrowstyle="->", lw=4, color="#bdc3c7", alpha=0.5))
ax3.text(0.98, (grid_top + grid_bottom)/2, "Depth\nProgression", ha="center", va="center", rotation=-90, fontsize=10, color="#7f8c8d", fontweight="bold")

# ── 7. Save ──────────────────────────────────────────────────────────────────
plt.suptitle(
    "HeadGenome Atlas: Anatomical Reorganization of Attention Heads",
    fontsize=18, fontweight="bold", y=1.02
)
plt.figtext(
    0.5, 0.95,
    f"N=1,568 | Kruskal–Wallis Spatial Enrichment: H={H:.1f}, p={p:.1e}, $\eta^2$={eta_sq:.3f}",
    ha="center", fontsize=12, color="#444444"
)

plt.tight_layout()
os.makedirs("outputs/final_artifacts", exist_ok=True)
out_path = "outputs/final_artifacts/headgenome_atlas.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")
