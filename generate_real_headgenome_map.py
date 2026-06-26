"""
generate_headgenome_map.py
==========================
Generates headgenome_map.png from the CANONICAL classification in
outputs/canonical_labels.json (produced by canonical_classification.py).

Fixes applied vs. previous version:
1. Single canonical data source — no more inconsistency between figures.
2. Relative depth Y-axis (0.0–1.0) for cross-architecture comparability.
3. Hollow-circle markers for Sink heads for visibility.
4. KDE bandwidth fixed via bw_method='scott' and curves normalised to [0, 1]
   so all classes are visually comparable.
5. Legend sample sizes derived from canonical_labels.json.
6. Kruskal-Wallis test reported from the same canonical data.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, kruskal
import os

# ── Load canonical labels ────────────────────────────────────────────────────
data   = json.load(open("outputs/canonical_labels.json"))
models = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]

# Build a flat record per head
rows = []
for model in models:
    m_data = data["models"][model]
    n_layers = m_data["n_layers"]
    for hid, info in m_data["heads"].items():
        label = info["label"]
        depth = info["relative_depth"]
        # split induction into Early/Late at depth 0.5
        if label == "induction":
            label = "Late Induction" if depth >= 0.5 else "Early Induction"
        elif label == "retrieval":
            label = "Retrieval"
        elif label == "local":
            label = "Local"
        elif label == "sink":
            label = "Sink"
        rows.append({"model": model, "depth": depth, "label": label})

# ── Counts ───────────────────────────────────────────────────────────────────
CLASSES  = ["Sink", "Local", "Retrieval", "Early Induction", "Late Induction"]
COLORS   = {
    "Sink":           "#9b59b6",
    "Local":          "#2ecc71",
    "Retrieval":      "#3498db",
    "Early Induction":"#f39c12",
    "Late Induction": "#e74c3c",
}

depths_by_class = {c: [] for c in CLASSES}
for r in rows:
    depths_by_class[r["label"]].append(r["depth"])

n_by_class = {c: len(v) for c, v in depths_by_class.items()}
print("Sample sizes:", n_by_class)
print(f"Total: {sum(n_by_class.values())}")

# ── Statistical test ─────────────────────────────────────────────────────────
groups = [np.array(depths_by_class[c]) for c in CLASSES]
H, p   = kruskal(*groups)
print(f"Kruskal-Wallis: H={H:.2f}, p={p:.2e}")
for c in CLASSES:
    d = np.array(depths_by_class[c])
    print(f"  {c:<18}: {np.mean(d):.3f} ± {np.std(d):.3f}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]}
)

np.random.seed(42)
for i, model in enumerate(models):
    model_rows = [r for r in rows if r["model"] == model]
    for label in CLASSES:
        pts = np.array([r["depth"] for r in model_rows if r["label"] == label])
        if len(pts) == 0:
            continue
        x = np.random.normal(i, 0.08, len(pts))
        if label == "Sink":
            ax1.scatter(x, pts, facecolors="none", edgecolors=COLORS[label],
                        s=90, linewidths=2, alpha=0.95, zorder=6)
        else:
            ax1.scatter(x, pts, c=COLORS[label], s=22, alpha=0.55,
                        edgecolors="none", zorder=2)

ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=12, fontweight="bold")
ax1.set_ylabel("Relative Network Depth", fontsize=12, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("Cross-Architecture Spatial Distribution", fontsize=13, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# ── KDE panel ────────────────────────────────────────────────────────────────
y_grid = np.linspace(-0.05, 1.05, 300)
for label in CLASSES:
    pts = np.array(depths_by_class[label])
    if len(pts) < 5:
        continue
    kde     = gaussian_kde(pts, bw_method="scott")
    density = kde(y_grid)
    density = density / density.max()          # normalise to [0, 1] for comparability
    ax2.plot(density, y_grid, color=COLORS[label], linewidth=2)
    ax2.fill_betweenx(y_grid, 0, density, color=COLORS[label], alpha=0.25)

ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(0, 1.25)
ax2.set_xlabel("Norm. Density", fontsize=11, fontweight="bold")
ax2.set_title("Global Density", fontsize=13, fontweight="bold")
ax2.set_yticks([])
ax2.grid(axis="x", linestyle="--", alpha=0.3)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_handles = []
for label in CLASSES:
    n = n_by_class[label]
    if label == "Sink":
        h = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                          markeredgecolor=COLORS[label], markersize=10,
                          markeredgewidth=2, label=f"{label} (n={n})")
    else:
        h = mpatches.Patch(facecolor=COLORS[label], label=f"{label} (n={n})")
    legend_handles.append(h)

ax1.legend(handles=legend_handles, loc="upper left",
           title="Head Taxonomy", fontsize=9, title_fontsize=10)

plt.suptitle(
    "HeadGenome Map: Spatial Distribution of Functional Attention Head Types\n"
    f"Across Transformer Architectures  "
    f"(Kruskal–Wallis H={H:.1f}, p={p:.1e})",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()

os.makedirs("outputs/final_artifacts", exist_ok=True)
plt.savefig("outputs/final_artifacts/headgenome_map.png", dpi=300, bbox_inches="tight")
print("\nSaved: outputs/final_artifacts/headgenome_map.png")
