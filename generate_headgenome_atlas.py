"""
generate_headgenome_atlas.py
============================
Creates the 3-panel iconic HeadGenome Atlas.

Panel A: Scatter plot (the evidence)
Panel B: Density plot + Trajectory Means (statistical summary)
Panel C: Simplified functional circuit (structural meaning)
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
        rows.append({"model": model, "depth": depth, "label": label})

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
eta_sq = (H - k + 1) / (total_n - k)  # Eta-squared effect size
print(f"Kruskal-Wallis: H={H:.2f}, p={p:.2e}, eta^2={eta_sq:.4f}")

# ── 3. Figure Layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9), facecolor="#f8f9fa")
gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.2, 1.5], wspace=0.15)

ax1 = fig.add_subplot(gs[0])  # Scatter
ax2 = fig.add_subplot(gs[1])  # Density
ax3 = fig.add_subplot(gs[2])  # Circuit

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
ax2.set_yticklabels([]) # Hide labels to stick right next to ax1
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
        x_val = 1.15  # Plot the means on the right side of the density plot
        traj_x.append(x_val)
        traj_y.append(y_val)
        ax2.scatter(x_val, y_val, color=COLORS[label], s=80, zorder=10, edgecolor="black")
        ax2.text(x_val + 0.05, y_val, f"Mean: {y_val:.2f}", va="center", fontsize=9, fontweight="bold", color=COLORS[label])

ax2.plot(traj_x, traj_y, color="black", linestyle="--", linewidth=2, alpha=0.6, zorder=5)

# ── 6. Panel C: Functional Circuit ───────────────────────────────────────────
ax3.axis("off")
ax3.set_title("C. Functional Maturation Circuit", fontsize=14, fontweight="bold", loc="left")

# Draw rectangles manually
def draw_box(ax, center, text, color, width=0.4, height=0.1):
    x, y = center
    box = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.02", ec="black", fc=color, alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", color="white", fontweight="bold", fontsize=12)

def draw_arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=2, color="gray", shrinkA=10, shrinkB=10))

# Coordinates
C_IN  = (0.5, 0.9)
C_LOC = (0.5, 0.7)
C_RET = (0.2, 0.45)
C_E_I = (0.8, 0.55)
C_L_I = (0.8, 0.35)
C_OUT = (0.5, 0.1)

ax3.text(C_IN[0], C_IN[1], "Input Sequence", ha="center", va="center", fontsize=12, fontweight="bold", color="#333333")
draw_box(ax3, C_LOC, "Local Precursor\n(Context Broadening)", COLORS["Local"])
draw_box(ax3, C_RET, "Retrieval\n(Semantic Match)", COLORS["Retrieval"], width=0.35)
draw_box(ax3, C_E_I, "Early Induction\n(Prefix Match)", COLORS["Early Induction"], width=0.35)
draw_box(ax3, C_L_I, "Late Induction\n(Payload Copy)", COLORS["Late Induction"], width=0.35)
ax3.text(C_OUT[0], C_OUT[1], "Output Predictions", ha="center", va="center", fontsize=12, fontweight="bold", color="#333333")

draw_arrow(ax3, C_IN, (C_LOC[0], C_LOC[1]+0.05))
draw_arrow(ax3, (C_LOC[0], C_LOC[1]-0.05), (C_RET[0], C_RET[1]+0.05))
draw_arrow(ax3, (C_LOC[0], C_LOC[1]-0.05), (C_E_I[0]-0.1, C_E_I[1]+0.05))
draw_arrow(ax3, (C_E_I[0], C_E_I[1]-0.05), (C_L_I[0], C_L_I[1]+0.05))
draw_arrow(ax3, (C_RET[0], C_RET[1]-0.05), (C_OUT[0]-0.05, C_OUT[1]+0.02))
draw_arrow(ax3, (C_L_I[0], C_L_I[1]-0.05), (C_OUT[0]+0.05, C_OUT[1]+0.02))

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.suptitle(
    "HeadGenome Atlas: Progressive Reorganization of Attention Heads",
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
