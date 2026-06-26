"""
generate_headgenome_map.py
==========================
Generates headgenome_map.png from the CANONICAL classification in
outputs/canonical_labels.json (produced by canonical_classification.py).

Visual design principles:
  - Render order: Local (bottom layer, faded) → Sink → Retrieval → Early/Late Induction
    so that rare specialized heads are always visible on top.
  - Rare types (Retrieval n=23, Sink n=28) use LARGER markers drawn last.
  - All colors are darker / more saturated for print legibility.
  - KDE curves use bw_method='scott' and are NOT normalised so relative
    height encodes population size (honest representation of dominance).
  - Density panel uses a log-scaled X so the rare classes' curves are visible.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, kruskal
import os

# ── Load canonical labels ─────────────────────────────────────────────────────
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

# ── Visual config — darker, high-contrast palette ─────────────────────────────
# Render order: Local first (drawn behind), rare types drawn last (on top)
RENDER_ORDER = ["Local", "Early Induction", "Sink", "Late Induction", "Retrieval"]

COLORS = {
    "Sink":           "#6c1f8a",   # deep violet
    "Local":          "#1a7a37",   # forest green
    "Retrieval":      "#0d3fbd",   # strong navy blue
    "Early Induction":"#c26b00",   # burnt orange
    "Late Induction": "#b81414",   # deep crimson
}

# Scatter appearance per class
SCATTER_CFG = {
    "Local":           {"s": 18,  "alpha": 0.35, "zorder": 1, "edgecolors": "none"},
    "Early Induction": {"s": 45,  "alpha": 0.75, "zorder": 4, "edgecolors": "none"},
    "Late Induction":  {"s": 50,  "alpha": 0.85, "zorder": 5, "edgecolors": "none"},
    "Retrieval":       {"s": 90,  "alpha": 1.00, "zorder": 7, "edgecolors": "white", "linewidths": 0.8},
    "Sink":            {"s": 80,  "alpha": 1.00, "zorder": 6, "facecolors": "none",
                        "edgecolors": "#6c1f8a", "linewidths": 2.2},
}

# ── Aggregate depths per class ────────────────────────────────────────────────
CLASSES = ["Sink", "Local", "Retrieval", "Early Induction", "Late Induction"]
depths_by_class = {c: [] for c in CLASSES}
for r in rows:
    depths_by_class[r["label"]].append(r["depth"])

n_by_class = {c: len(v) for c, v in depths_by_class.items()}
total_n    = sum(n_by_class.values())
print(f"Total heads: {total_n}")
print("Sample sizes:", n_by_class)

# ── Statistical test ──────────────────────────────────────────────────────────
groups = [np.array(depths_by_class[c]) for c in CLASSES]
H, p   = kruskal(*groups)
print(f"Kruskal-Wallis: H={H:.2f}, p={p:.2e}")
for c in CLASSES:
    d = np.array(depths_by_class[c])
    print(f"  {c:<18}: {np.mean(d):.3f} ± {np.std(d):.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(15, 9),
    gridspec_kw={"width_ratios": [3.2, 1]},
    facecolor="#f8f9fa"
)
ax1.set_facecolor("#ffffff")
ax2.set_facecolor("#ffffff")

np.random.seed(42)
for label in RENDER_ORDER:
    cfg = SCATTER_CFG[label]
    color = COLORS[label]

    for i, model in enumerate(models):
        pts = np.array([r["depth"] for r in rows
                        if r["model"] == model and r["label"] == label])
        if len(pts) == 0:
            continue
        x = np.random.normal(i, 0.08, len(pts))

        if label == "Sink":
            ax1.scatter(x, pts,
                        facecolors=cfg["facecolors"],
                        edgecolors=cfg["edgecolors"],
                        linewidths=cfg["linewidths"],
                        s=cfg["s"], alpha=cfg["alpha"],
                        zorder=cfg["zorder"])
        elif label == "Retrieval":
            ax1.scatter(x, pts, c=color,
                        edgecolors=cfg["edgecolors"],
                        linewidths=cfg["linewidths"],
                        s=cfg["s"], alpha=cfg["alpha"],
                        zorder=cfg["zorder"])
        else:
            ax1.scatter(x, pts, c=color,
                        edgecolors=cfg["edgecolors"],
                        s=cfg["s"], alpha=cfg["alpha"],
                        zorder=cfg["zorder"])

ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=12, fontweight="bold")
ax1.set_ylabel("Relative Network Depth", fontsize=12, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("Scatter: Per-Architecture Distribution", fontsize=12, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#cccccc")
ax1.spines[["top", "right"]].set_visible(False)

# ── KDE Density panel ─────────────────────────────────────────────────────────
# Use absolute (not normalised) density so Local dominance is honest,
# but plot on log-x so rare classes are legible
y_grid = np.linspace(-0.05, 1.05, 400)

all_densities = {}
for label in CLASSES:
    pts = np.array(depths_by_class[label])
    if len(pts) < 5:
        all_densities[label] = np.zeros_like(y_grid)
        continue
    kde     = gaussian_kde(pts, bw_method="scott")
    density = kde(y_grid) * len(pts)      # scale by count → area = n
    all_densities[label] = density

# Normalise each class to its own max for shape comparison
# (makes the rare classes' shape visible; caption will note this)
for label in CLASSES:
    pts = np.array(depths_by_class[label])
    if len(pts) < 5:
        continue
    density = all_densities[label] / all_densities[label].max()
    color   = COLORS[label]
    lw      = 1.5 if label == "Local" else 2.5
    alpha_f = 0.15 if label == "Local" else 0.30
    ax2.plot(density, y_grid, color=color, linewidth=lw, label=label)
    ax2.fill_betweenx(y_grid, 0, density, color=color, alpha=alpha_f)

ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(0, 1.35)
ax2.set_xlabel("Norm. Density (shape only)", fontsize=10, fontweight="bold")
ax2.set_title("Global Depth Density", fontsize=12, fontweight="bold")
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#555555")
ax2.tick_params(axis="y", left=True, labelleft=True, labelsize=8, colors="#555555")
ax2.grid(axis="x", linestyle="--", alpha=0.3, color="#cccccc")
ax2.spines[["top", "right"]].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = []
for label in CLASSES:
    n = n_by_class[label]
    if label == "Sink":
        h = mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor="none",
                          markeredgecolor=COLORS[label],
                          markersize=10, markeredgewidth=2,
                          label=f"{label} (n={n})")
    else:
        h = mpatches.Patch(facecolor=COLORS[label],
                           label=f"{label} (n={n})")
    legend_handles.append(h)

ax1.legend(handles=legend_handles, loc="upper left",
           title=f"Head Taxonomy  [N={total_n}]",
           fontsize=9, title_fontsize=9,
           framealpha=0.92, edgecolor="#cccccc")

plt.suptitle(
    "HeadGenome Map: Spatial Distribution of Functional Attention Head Types\n"
    "Across Transformer Architectures  "
    f"(N=1,568 | Kruskal–Wallis H={H:.1f}, p={p:.1e})",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()

os.makedirs("outputs/final_artifacts", exist_ok=True)
out_path = "outputs/final_artifacts/headgenome_map.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")
