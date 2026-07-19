"""
generate_headline_figure.py
============================
Generates outputs/headline_figure.png — a publication-quality, light-mode
headline research poster figure summarising all 6 sections of the
Transformer Trajectory study.

Run from the project root:
    python code/generate_headline_figure.py
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

OUT = Path("outputs/headline_figure.png")
ROOT = Path("outputs")

# ═══════════════════════════════════════════════════════
# 1. LOAD ALL REAL DATA
# ═══════════════════════════════════════════════════════

# ---------- §1  Probing accuracy (from SECTION1_REPORT JSON) -----------
# (these are the exact values from outputs/probing/ JSON)
PROBE = {
    "Qwen2.5-1.5B": [
        0.4333, 0.6889, 0.6889, 0.7000, 0.7056, 0.7944, 0.7333, 0.7111,
        0.7889, 0.7889, 0.8111, 0.8111, 0.8056, 0.8778, 0.8722, 0.8778,
        0.9000, 0.8778, 0.8778, 0.8778, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000,
    ],
    "Llama-3.2-1B": [
        0.6611, 0.7222, 0.7611, 0.8111, 0.8111, 0.8444, 0.8778, 0.8778,
        0.9389, 0.9111, 0.9556, 0.9556, 0.9556, 0.9556, 0.9556, 0.9722,
    ],
    "Phi-1.5": [
        0.6889, 0.7222, 0.7889, 0.7778, 0.8111, 0.8611, 0.9056, 0.9389,
        0.9389, 0.9389, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    ],
}
PROBE_SHUFFLE = 0.30   # ~95th-pct shuffle line (constant across all models)

# ---------- §2  F-ratio  (loaded from JSON) ----------------------------
f_data = json.loads((ROOT / "intra_mapping" / "f_statistic_data.json").read_text())
FRATIO = {m: np.array(f_data[m]["real_F"]) for m in f_data}
FSHUFFLE = {m: np.array(f_data[m]["shuffle_95th"]) for m in f_data}

# ---------- §3  Cross-arch confusion matrices --------------------------
dtw_raw = json.loads((ROOT / "dtw_results" / "dtw_cross_arch_alignment.json").read_text())
CATS = ["arithmetic", "comparison", "copy", "counting", "fact_recall", "sorting"]

def _to_matrix(pair_key):
    m = dtw_raw[pair_key]["true_matrix"]
    return np.array([[m[r][c] for c in CATS] for r in CATS])

CM_QL = _to_matrix("Qwen2.5-1.5B_vs_Llama-3.2-1B")
CM_QP = _to_matrix("Qwen2.5-1.5B_vs_phi-1_5")
CM_LP = _to_matrix("Llama-3.2-1B_vs_phi-1_5")
# Average across 3 pairs for a single summary heatmap
CM_AVG = (CM_QL / CM_QL.max() + CM_QP / CM_QP.max() + CM_LP / CM_LP.max()) / 3.0

# ---------- §4  Causal intervention  (loaded from JSON) ----------------
caus_raw = json.loads((ROOT / "causal_intervention" / "sweep_results.json").read_text())

def _hijack_series(pair_key, c="5.0"):
    pair = caus_raw[pair_key]
    real = np.zeros(28)
    rand = np.zeros(28)
    for l in range(28):
        entry = pair.get(str(l), {})
        if c in entry:
            real[l] = entry[c]["b_rate_real"]
            rand[l] = entry[c]["b_rate_rand"]
    return real, rand

FC_REAL, FC_RAND = _hijack_series("fact_recall_to_comparison", c="5.0")
AS_REAL, AS_RAND = _hijack_series("arithmetic_to_sorting", c="1.5")

# ---------- §5  DTA transition shares (loaded from JSON) ---------------
dta_raw = json.loads((ROOT / "generator_analysis" / "dta_results.json").read_text())
DTA_TRANS_MLP   = np.array(dta_raw["trans_mlps"])      # [28]
DTA_TRANS_HEADS = np.array(dta_raw["trans_heads"])     # [28, n_heads]
# Rise phase: layers 10-20
RISE = slice(10, 21)
DTA_MLP_RISE  = DTA_TRANS_MLP[RISE]                    # per-layer MLP %
# Per-layer minimum (most negative) head = opponent
DTA_OPP_RISE  = DTA_TRANS_HEADS[RISE].min(axis=1) * 100

# ---------- §6  DLA mass distribution (from report numbers) ------------
# Exact values reported in Section 6 report
DLA_COMPS = [
    ("L27 MLP", 36.75, "mlp"),
    ("L26 MLP", 13.44, "mlp"),
    ("L22 MLP",  9.50, "mlp"),
    ("L25 MLP",  9.19, "mlp"),
    ("L24 MLP",  9.06, "mlp"),
    ("L23 MLP",  6.72, "mlp"),
    ("L21 MLP",  4.44, "mlp"),
    ("L18 MLP",  4.03, "mlp"),
    ("L27 H10",  3.78, "attn"),
    ("L20 MLP",  3.72, "mlp"),
]
DLA_TOTAL_ABS = 193.38   # from report
DLA_TOP3  = 59.69
DLA_TOP10 = 100.62

# ═══════════════════════════════════════════════════════
# 2. COLOUR PALETTE  (light-mode, publication-ready)
# ═══════════════════════════════════════════════════════
C = {
    "s1":    "#5B4FCF",   # §1 indigo
    "s2":    "#0077B6",   # §2 ocean
    "s3":    "#0A9F6E",   # §3 teal-green
    "s4":    "#D97706",   # §4 amber/warning
    "s5":    "#7C3AED",   # §5 purple
    "s6":    "#B91C1C",   # §6 crimson
    "llama": "#0077B6",
    "phi":   "#0A9F6E",
    "qwen":  "#5B4FCF",
    "null":  "#9CA3AF",
    "mlp":   "#6D28D9",
    "attn":  "#DC2626",
    "bg":    "#FAFAFA",
    "panel": "#FFFFFF",
    "grid":  "#E5E7EB",
    "text":  "#111827",
    "muted": "#6B7280",
    "diag":  "#065F46",  # deep green for matrix diagonal
    "offdiag": "#9F1239",
}

# ═══════════════════════════════════════════════════════
# 3. LAYOUT
# ═══════════════════════════════════════════════════════
FIG_W, FIG_H = 24, 16   # inches at 150 dpi → 3600 × 2400 px
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C["bg"])

# Outer grid: 3 rows
outer = gridspec.GridSpec(
    3, 1, figure=fig,
    hspace=0.46,
    top=0.93, bottom=0.05, left=0.035, right=0.978,
)

# Row 0 — Header / title band
ax_title = fig.add_subplot(outer[0])
ax_title.set_axis_off()

# Row 1 — 6 section panels
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 6, subplot_spec=outer[1], wspace=0.55,
)

# Row 2 — 3 synthesis panels
row2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[2], wspace=0.45,
)

def panel(gs_loc, *, label, title, color, subtitle=None):
    ax = fig.add_subplot(gs_loc)
    ax.set_facecolor(C["panel"])
    ax.tick_params(labelsize=7, colors=C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    ax.xaxis.label.set_color(C["muted"])
    for sp in ax.spines.values():
        sp.set_color(C["grid"])
    # top colour stripe
    ax.axhspan(ax.get_ylim()[1], ax.get_ylim()[1], xmin=0, xmax=1,
               color=color, lw=0)
    # Section label + title
    ax.set_title(
        f"§{label}  {title}", fontsize=8.5, fontweight="bold",
        color=C["text"], pad=6, loc="left",
    )
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=6.5, color=C["muted"], style="italic", va="bottom")
    return ax

def draw_top_stripe(ax, color, height_frac=0.025):
    """Draw a thin colour stripe at the top of an axes."""
    ax.axhspan(
        ax.get_ylim()[1] * (1 - height_frac), ax.get_ylim()[1],
        color=color, alpha=1, zorder=10, lw=0,
    )

def annotate_finding(ax, text, fontsize=6.5, y=-0.22):
    ax.text(
        0.5, y, text, transform=ax.transAxes,
        fontsize=fontsize, ha="center", va="top",
        color=C["text"], wrap=True,
        bbox=dict(fc="#F3F4F6", ec=C["grid"], lw=0.6, boxstyle="round,pad=0.35"),
    )

# ═══════════════════════════════════════════════════════
# 4.  HEADER
# ═══════════════════════════════════════════════════════
ax_title.text(
    0.5, 0.82,
    "Transformer Computation: Trajectories, Builders & Readouts",
    transform=ax_title.transAxes,
    fontsize=20, fontweight="black", ha="center", va="top", color=C["text"],
)
ax_title.text(
    0.5, 0.42,
    "A 6-Section Mechanistic Investigation into the Geometry and Neural Architecture of Cognitive Operations in LLMs  ·  "
    "Qwen2.5-1.5B · Llama-3.2-1B · Phi-1.5",
    transform=ax_title.transAxes,
    fontsize=9, ha="center", va="top", color=C["muted"],
)
# Section flow pills
sections = [
    ("§1 Emergence",       C["s1"]),
    ("§2 Dynamics",        C["s2"]),
    ("§3 Cross-Arch",      C["s3"]),
    ("§4 Causal ✗",        C["s4"]),
    ("§5 Builders",        C["s5"]),
    ("§6 Readout",         C["s6"]),
]
n = len(sections)
for i, (lbl, col) in enumerate(sections):
    x = 0.07 + i * 0.15
    ax_title.text(
        x, 0.05, lbl, transform=ax_title.transAxes,
        fontsize=8.5, fontweight="bold", ha="center", va="center",
        color="white",
        bbox=dict(fc=col, ec="none", boxstyle="round,pad=0.45"),
    )
    if i < n - 1:
        ax_title.annotate(
            "", xy=(x + 0.13, 0.05), xytext=(x + 0.07, 0.05),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1.2),
        )

# ═══════════════════════════════════════════════════════
# 5.  §1  Probe accuracy
# ═══════════════════════════════════════════════════════
ax1 = fig.add_subplot(row1[0])
ax1.set_facecolor(C["panel"])
ax1.set_title("§1  Task Emergence", fontsize=8.5, fontweight="bold",
              color=C["text"], pad=5, loc="left")
ax1.text(0.0, 1.02, "Linear probe accuracy vs. layer depth",
         transform=ax1.transAxes, fontsize=6.5, color=C["muted"], style="italic")

model_styles = {
    "Qwen2.5-1.5B": (C["qwen"], 2.2, "-",  "Qwen"),
    "Llama-3.2-1B": (C["llama"], 1.5, "--", "Llama"),
    "Phi-1.5":      (C["phi"],  1.5, "-.", "Phi"),
}
for mname, (col, lw, ls, lbl) in model_styles.items():
    ys = PROBE[mname]
    xs = np.arange(len(ys))
    ax1.plot(xs, ys, color=col, lw=lw, ls=ls, label=lbl, zorder=3)

ax1.axhline(PROBE_SHUFFLE, color=C["null"], lw=1, ls=":", label="Chance (95th)")
ax1.fill_between(np.arange(28), PROBE_SHUFFLE, alpha=0.06, color=C["null"])
ax1.set_xlim(0, 27); ax1.set_ylim(0.0, 1.07)
ax1.set_xlabel("Layer", fontsize=7, color=C["muted"])
ax1.set_ylabel("Accuracy", fontsize=7, color=C["muted"])
ax1.tick_params(labelsize=6.5)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax1.grid(axis="y", color=C["grid"], lw=0.5, zorder=0)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
leg1 = ax1.legend(fontsize=6.5, loc="lower right", frameon=True,
                  framealpha=0.9, edgecolor=C["grid"])
# Onset annotations
ax1.axvline(20, color=C["qwen"], lw=0.8, ls=":", alpha=0.6)
ax1.axvline(10, color=C["phi"],  lw=0.8, ls=":", alpha=0.6)
ax1.text(20, 0.55, "L20\nQwen", fontsize=6, color=C["qwen"], ha="right")
ax1.text(10, 0.55, "L10\nPhi",  fontsize=6, color=C["phi"],  ha="left")
annotate_finding(
    ax1,
    "Monotonic emergence across all 3 architectures.\nOnset layer unstable (±3 L); pattern is the claim.",
    y=-0.28,
)
# Colour stripe
ax1.add_patch(mpatches.FancyBboxPatch(
    (0, 1.0), 1, 0.04, transform=ax1.transAxes, clip_on=False,
    fc=C["s1"], ec="none", boxstyle="square,pad=0",
))

# ═══════════════════════════════════════════════════════
# 6.  §2  F-ratio Dynamics
# ═══════════════════════════════════════════════════════
ax2 = fig.add_subplot(row1[1])
ax2.set_facecolor(C["panel"])
ax2.set_title("§2  Trajectory Dynamics", fontsize=8.5, fontweight="bold",
              color=C["text"], pad=5, loc="left")
ax2.text(0.0, 1.02, "Between-category F-ratio vs. layer",
         transform=ax2.transAxes, fontsize=6.5, color=C["muted"], style="italic")

f_styles = {
    "Qwen2.5-1.5B": (C["qwen"], 2.2, "-",  "Qwen"),
    "Llama-3.2-1B": (C["llama"], 1.5, "--", "Llama"),
    "phi-1_5":       (C["phi"],  1.5, "-.", "Phi"),
}
for mkey, (col, lw, ls, lbl) in f_styles.items():
    ys = FRATIO[mkey]
    xs = np.arange(len(ys))
    ax2.plot(xs, ys, color=col, lw=lw, ls=ls, label=lbl, zorder=3)
    shuf = FSHUFFLE[mkey]
    ax2.fill_between(xs, shuf, ys, where=ys > shuf,
                     alpha=0.08, color=col, zorder=1)

# Null lines (plot for Qwen as reference)
ax2.plot(np.arange(28), FSHUFFLE["Qwen2.5-1.5B"],
         color=C["null"], lw=1, ls=":", label="Null (shuffle)")

# Peak annotation for Qwen
qp = int(np.argmax(FRATIO["Qwen2.5-1.5B"]))
ax2.axvline(qp, color=C["s4"], lw=0.9, ls="--", alpha=0.7)
ax2.text(qp + 0.3, FRATIO["Qwen2.5-1.5B"].max() * 0.92,
         f"Peak\nL{qp}", fontsize=6, color=C["s4"])

# Scatter zone
scatter_start = 22
ax2.axvspan(scatter_start, 27, alpha=0.06, color=C["s6"])
ax2.text(25, 1.5, "Lexical\nScatter", fontsize=6, ha="center",
         color=C["s6"], fontweight="bold")

ax2.set_xlabel("Layer", fontsize=7, color=C["muted"])
ax2.set_ylabel("F-Ratio (log)", fontsize=7, color=C["muted"])
ax2.set_yscale("log")
ax2.set_xlim(0, 27)
ax2.tick_params(labelsize=6.5)
ax2.grid(axis="y", color=C["grid"], lw=0.5, zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=6.5, loc="lower right", frameon=True,
           framealpha=0.9, edgecolor=C["grid"])
annotate_finding(
    ax2,
    "Trunk-&-Branch geometry peaks at L21 (Qwen).\nFinal layers scatter (SSW +7×, SSB +5×).",
    y=-0.28,
)
ax2.add_patch(mpatches.FancyBboxPatch(
    (0, 1.0), 1, 0.04, transform=ax2.transAxes, clip_on=False,
    fc=C["s2"], ec="none", boxstyle="square,pad=0",
))

# ═══════════════════════════════════════════════════════
# 7.  §3  Cross-arch confusion matrix (averaged)
# ═══════════════════════════════════════════════════════
ax3 = fig.add_subplot(row1[2])
ax3.set_facecolor(C["panel"])
ax3.set_title("§3  Cross-Architecture", fontsize=8.5, fontweight="bold",
              color=C["text"], pad=5, loc="left")
ax3.text(0.0, 1.02, "Avg DTW cost (6×6), Qwen/Llama/Phi pairs",
         transform=ax3.transAxes, fontsize=6.5, color=C["muted"], style="italic")

# Custom colormap: white→deep-red (off-diag = high cost = bad)
cmap_confusion = LinearSegmentedColormap.from_list(
    "conf", ["#f0fdf4", "#bbf7d0", "#4ade80", "#15803d",
             "#fef2f2", "#fecaca", "#ef4444", "#7f1d1d"],
    N=256,
)
# For display we want diagonal = low = GREEN, off-diag = high = RED
# Build a diverging view: normalise across all values, diagonal will be low
im3 = ax3.imshow(CM_AVG, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

short_cats = ["Arith", "Comp", "Copy", "Count", "Fact", "Sort"]
ax3.set_xticks(range(6)); ax3.set_xticklabels(short_cats, fontsize=6, rotation=35, ha="right")
ax3.set_yticks(range(6)); ax3.set_yticklabels(short_cats, fontsize=6)

# Highlight diagonal
for i in range(6):
    ax3.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="#065F46", lw=2, zorder=5))
    ax3.text(i, i, f"{CM_AVG[i, i]:.2f}",
             ha="center", va="center", fontsize=6.5,
             fontweight="bold", color="white",
             path_effects=[pe.withStroke(linewidth=1.5, foreground="#065F46")])

plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04,
             label="Norm. DTW cost").ax.tick_params(labelsize=6)
annotate_finding(
    ax3,
    "Diagonal = global min across all 720 permutations.\np = 1/720 ≈ 0.00139 · replicated for all 3 pairs.",
    y=-0.28,
)
ax3.add_patch(mpatches.FancyBboxPatch(
    (0, 1.0), 1, 0.04, transform=ax3.transAxes, clip_on=False,
    fc=C["s3"], ec="none", boxstyle="square,pad=0",
))

# ═══════════════════════════════════════════════════════
# 8.  §4  Causal intervention  — hijack rate
# ═══════════════════════════════════════════════════════
ax4 = fig.add_subplot(row1[3])
ax4.set_facecolor(C["panel"])
ax4.set_title("§4  Causal Steering (FALSIFIED)", fontsize=8.5, fontweight="bold",
              color=C["s4"], pad=5, loc="left")
ax4.text(0.0, 1.02, "Hijack B-rate vs. layer  (c = 5.0 / 1.5)",
         transform=ax4.transAxes, fontsize=6.5, color=C["muted"], style="italic")

layers28 = np.arange(28)
# F-ratio normalised overlay
fratio_norm = (FRATIO["Qwen2.5-1.5B"] - FRATIO["Qwen2.5-1.5B"].min()) / \
              (FRATIO["Qwen2.5-1.5B"].max() - FRATIO["Qwen2.5-1.5B"].min())
ax4.fill_between(layers28, 0, fratio_norm, alpha=0.08, color=C["s2"], label="F-Ratio (norm.)")
ax4.plot(layers28, fratio_norm, color=C["s2"], lw=1, ls="--", alpha=0.6)

ax4.plot(layers28, FC_REAL, color=C["s4"], lw=2, marker="o", ms=3.5,
         label="Fact→Comp (real)")
ax4.plot(layers28, FC_RAND, color=C["s4"], lw=0.8, ls=":", marker="x", ms=3,
         alpha=0.6, label="Fact→Comp (rand)")
ax4.plot(layers28, AS_REAL, color=C["s5"], lw=2, marker="s", ms=3.5,
         label="Arith→Sort (real)")
ax4.plot(layers28, AS_RAND, color=C["s5"], lw=0.8, ls=":", marker="x", ms=3,
         alpha=0.6, label="Arith→Sort (rand)")

# Spike annotations
spike_fc = int(np.argmax(FC_REAL[:20]))
spike_as = int(np.argmax(AS_REAL))
ax4.annotate(f"L{spike_fc}\n{FC_REAL[spike_fc]:.0%}",
             xy=(spike_fc, FC_REAL[spike_fc]),
             xytext=(spike_fc + 2.5, FC_REAL[spike_fc] - 0.12),
             fontsize=6.5, color=C["s4"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C["s4"], lw=0.9))
ax4.annotate(f"L{spike_as}\n{AS_REAL[spike_as]:.0%}",
             xy=(spike_as, AS_REAL[spike_as]),
             xytext=(spike_as - 6, AS_REAL[spike_as] + 0.08),
             fontsize=6.5, color=C["s5"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C["s5"], lw=0.9))

# OOD collapse zone
ax4.axvspan(14, 27, alpha=0.04, color=C["s6"])
ax4.text(21, 0.88, "OOD: 0%\n(collapsed)", fontsize=6,
         ha="center", color=C["s6"], fontweight="bold")

ax4.set_xlim(0, 27); ax4.set_ylim(-0.05, 1.05)
ax4.set_xlabel("Layer", fontsize=7, color=C["muted"])
ax4.set_ylabel("Hijack B-rate", fontsize=7, color=C["muted"])
ax4.tick_params(labelsize=6.5)
ax4.grid(axis="y", color=C["grid"], lw=0.5)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
ax4.legend(fontsize=5.5, loc="upper left", ncol=2, frameon=True,
           framealpha=0.9, edgecolor=C["grid"])
annotate_finding(
    ax4,
    "Full-curve r ≈ 0.05. Spikes are brittle overfits or degenerate\ntoken-collapse (16.2% unique outputs). OOD: 0%.",
    y=-0.28,
)
ax4.add_patch(mpatches.FancyBboxPatch(
    (0, 1.0), 1, 0.04, transform=ax4.transAxes, clip_on=False,
    fc=C["s4"], ec="none", boxstyle="square,pad=0",
))

# ═══════════════════════════════════════════════════════
# 9.  §5  DTA Transition shares
# ═══════════════════════════════════════════════════════
ax5 = fig.add_subplot(row1[4])
ax5.set_facecolor(C["panel"])
ax5.set_title("§5  Generator Analysis", fontsize=8.5, fontweight="bold",
              color=C["text"], pad=5, loc="left")
ax5.text(0.0, 1.02, "Dual-Metric DTA transition shares (Rise L10–20)",
         transform=ax5.transAxes, fontsize=6.5, color=C["muted"], style="italic")

rise_layers = np.arange(10, 21)
bar_w = 0.4

bars_mlp = ax5.bar(
    rise_layers - bar_w / 2, DTA_MLP_RISE, bar_w,
    color=C["mlp"], alpha=0.85, label="MLP (builder)", zorder=3,
)
bars_opp = ax5.bar(
    rise_layers + bar_w / 2, DTA_OPP_RISE, bar_w,
    color=C["attn"], alpha=0.75, label="Attn (opponent)", zorder=3,
)

ax5.axhline(0, color=C["text"], lw=0.8, zorder=2)
ax5.axhline(100, color=C["mlp"], lw=0.7, ls="--", alpha=0.5)
ax5.text(10.1, 102, "100%", fontsize=6, color=C["mlp"])

# Annotate the >100% bars
for bar in bars_mlp:
    h = bar.get_height()
    if h > 100:
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                 f"{h:.0f}%", ha="center", fontsize=6,
                 color=C["mlp"], fontweight="bold")

ax5.set_xlabel("Layer", fontsize=7, color=C["muted"])
ax5.set_ylabel("Transition share (%)", fontsize=7, color=C["muted"])
ax5.set_xticks(rise_layers[::2])
ax5.tick_params(labelsize=6.5)
ax5.set_ylim(-30, 125)
ax5.grid(axis="y", color=C["grid"], lw=0.5)
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
ax5.legend(fontsize=6.5, loc="upper right", frameon=True,
           framealpha=0.9, edgecolor=C["grid"])
annotate_finding(
    ax5,
    "MLPs: ~90% transition share. All top-10 opponents\nare Attention Heads. Replicated on Fact/Comp pair.",
    y=-0.28,
)
ax5.add_patch(mpatches.FancyBboxPatch(
    (0, 1.0), 1, 0.04, transform=ax5.transAxes, clip_on=False,
    fc=C["s5"], ec="none", boxstyle="square,pad=0",
))

# ═══════════════════════════════════════════════════════
# 10.  §6  DLA mass distribution
# ═══════════════════════════════════════════════════════
ax6 = fig.add_subplot(row1[5])
ax6.set_facecolor(C["panel"])
ax6.set_title("§6  Control Signals", fontsize=8.5, fontweight="bold",
              color=C["text"], pad=5, loc="left")
ax6.text(0.0, 1.02, "Top-10 DLA components (absolute mass)",
         transform=ax6.transAxes, fontsize=6.5, color=C["muted"], style="italic")

labels6 = [c[0] for c in DLA_COMPS]
vals6   = [c[1] for c in DLA_COMPS]
types6  = [c[2] for c in DLA_COMPS]
colors6 = [C["mlp"] if t == "mlp" else C["attn"] for t in types6]
y_pos   = np.arange(len(labels6))[::-1]

for i, (lbl, val, col, yp) in enumerate(zip(labels6, vals6, colors6, y_pos)):
    ax6.barh(yp, val, color=col, alpha=0.82, height=0.65, zorder=3)
    ax6.text(val + 0.3, yp, f"{val:.1f}", va="center", fontsize=6.5,
             color=C["text"])

# Top-3 and Top-10 cumulative markers
top3_x  = sum(vals6[:3]) / DLA_TOTAL_ABS * 100 * max(vals6) / 100
top10_x = sum(vals6[:10]) / DLA_TOTAL_ABS * 100 * max(vals6) / 100

ax6.set_yticks(y_pos)
ax6.set_yticklabels(labels6, fontsize=6.5)
ax6.set_xlabel("|DLA| score", fontsize=7, color=C["muted"])
ax6.tick_params(labelsize=6.5)
ax6.set_xlim(0, max(vals6) * 1.18)
ax6.grid(axis="x", color=C["grid"], lw=0.5)
ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

# Legend patches
leg_mlp  = mpatches.Patch(color=C["mlp"],  alpha=0.82, label="MLP")
leg_attn = mpatches.Patch(color=C["attn"], alpha=0.82, label="Attn Head")
ax6.legend(handles=[leg_mlp, leg_attn], fontsize=6.5, loc="lower right",
           frameon=True, framealpha=0.9, edgecolor=C["grid"])

ax6.text(0.5, -0.23,
         f"Top-3: {DLA_TOP3/DLA_TOTAL_ABS*100:.1f}%  |  Top-10: {DLA_TOP10/DLA_TOTAL_ABS*100:.1f}%  →  H4 Favoured",
         transform=ax6.transAxes, fontsize=7, ha="center",
         color=C["s6"], fontweight="bold")
annotate_finding(
    ax6,
    "Ablating top-3 DLA heads: 0% accuracy drop (H3 falsified).\nRandom MLP ablation also 0% → generic fragility.",
    y=-0.38,
)
ax6.add_patch(mpatches.FancyBboxPatch(
    (0, 1.0), 1, 0.04, transform=ax6.transAxes, clip_on=False,
    fc=C["s6"], ec="none", boxstyle="square,pad=0",
))

# ═══════════════════════════════════════════════════════
# 11.  ROW 2 — Synthesis panels
# ═══════════════════════════════════════════════════════

# ── 11a.  Mechanism diagram (residual stream river) ──
ax_m = fig.add_subplot(row2[0])
ax_m.set_facecolor(C["panel"])
ax_m.set_title("The Mechanism (Simplified)", fontsize=9, fontweight="bold",
               color=C["text"], pad=5)
ax_m.set_xlim(0, 28); ax_m.set_ylim(-1.2, 2.2)
ax_m.axis("off")

def _lerp(a, b, t):
    return a + (b - a) * t

# Draw river (residual stream band)
stream_y, stream_h = 0.4, 0.5
for i in range(28):
    frac = i / 27
    # Colour transitions from indigo → purple → red as task solidifies
    col_r = _lerp(91, 124, frac)
    col_g = _lerp(79, 58, frac)
    col_b = _lerp(207, 58, min(frac * 2, 1))
    rgba = (col_r / 255, col_g / 255, col_b / 255, 0.3 + 0.5 * frac)
    rect = plt.Rectangle((i, stream_y), 1, stream_h, fc=rgba, ec="none", zorder=1)
    ax_m.add_patch(rect)

# MLP arrows (upward, builders)
for l in [10, 12, 14, 16, 18, 20]:
    strength = 1.0 if 10 <= l <= 20 else 0.5
    ax_m.annotate("", xy=(l + 0.5, stream_y), xytext=(l + 0.5, stream_y - 0.6 * strength),
                  arrowprops=dict(arrowstyle="->", color=C["mlp"],
                                  lw=1.4 * strength, alpha=0.8))
    if l <= 18:
        ax_m.text(l + 0.5, stream_y - 0.75, "MLP", ha="center",
                  fontsize=5.5, color=C["mlp"], fontweight="bold")

# Attention drag arrows (downward)
for l in [11, 13, 15, 17, 19]:
    ax_m.annotate("", xy=(l + 0.5, stream_y + stream_h),
                  xytext=(l + 0.5, stream_y + stream_h + 0.45),
                  arrowprops=dict(arrowstyle="->", color=C["attn"],
                                  lw=1.0, alpha=0.7))
    ax_m.text(l + 0.5, stream_y + stream_h + 0.55, "Attn",
              ha="center", fontsize=5.5, color=C["attn"])

# Branching at L20
bx = 20.5
ax_m.annotate("Arithmetic", xy=(27, stream_y + 0.15), xytext=(bx, stream_y + stream_h / 2),
              fontsize=7, color=C["s5"], fontweight="bold",
              arrowprops=dict(arrowstyle="-|>", color=C["s5"], lw=1.3,
                              connectionstyle="arc3,rad=-0.25"))
ax_m.annotate("Sorting", xy=(27, stream_y + 0.75), xytext=(bx, stream_y + stream_h / 2),
              fontsize=7, color=C["s2"], fontweight="bold",
              arrowprops=dict(arrowstyle="-|>", color=C["s2"], lw=1.3,
                              connectionstyle="arc3,rad=0.25"))

# IN/OUT boxes
ax_m.add_patch(plt.Rectangle((0, stream_y - 0.05), 1.2, stream_h + 0.1,
               fc="#EEF2FF", ec=C["s1"], lw=1.5, zorder=4))
ax_m.text(0.6, stream_y + stream_h / 2, "IN", ha="center", va="center",
          fontsize=8, fontweight="bold", color=C["s1"], zorder=5)
ax_m.text(2, 1.9, "← Residual Stream Depth →", fontsize=7,
          color=C["muted"])
ax_m.add_patch(plt.Rectangle((26.5, stream_y - 0.05), 1.4, stream_h + 0.1,
               fc="#FEF2F2", ec=C["s6"], lw=1.5, zorder=4))
ax_m.text(27.2, stream_y + stream_h / 2, "OUT", ha="center", va="center",
          fontsize=7, fontweight="bold", color=C["s6"], zorder=5)

# Layer ticks
for l in range(0, 28, 4):
    ax_m.text(l + 0.5, stream_y + stream_h + 0.1, f"L{l}",
              ha="center", fontsize=5.5, color=C["muted"])

ax_m.text(14, -1.0,
          "MLPs push residual stream forward along each trajectory branch.\n"
          "Attention Heads drag backward — acting as trajectory opponents.",
          ha="center", fontsize=7, color=C["text"], style="italic",
          va="top")

# ── 11b. Hypothesis scorecard ──
ax_h = fig.add_subplot(row2[1])
ax_h.set_facecolor(C["panel"])
ax_h.set_title("Hypothesis Scorecard", fontsize=9, fontweight="bold",
               color=C["text"], pad=5)
ax_h.axis("off")

scorecard = [
    ("Trajectories exist & are conserved",    "CONFIRMED",  "#065F46", "#D1FAE5"),
    ("Causal power tracks geometry (F-ratio)", "FALSIFIED",  "#7F1D1D", "#FEE2E2"),
    ("MLPs build trajectory geometry",         "CONFIRMED",  "#065F46", "#D1FAE5"),
    ("Attn Heads build geometry",              "FALSIFIED",  "#7F1D1D", "#FEE2E2"),
    ("H1: Trajectory injection sufficient",    "FALSIFIED",  "#7F1D1D", "#FEE2E2"),
    ("H2/H3: Specific readout head triggers",  "FALSIFIED",  "#7F1D1D", "#FEE2E2"),
    ("H4: Distributed readout circuit",        "FAVOURED",   "#78350F", "#FEF3C7"),
    ("DLA/ablation isolates true circuit",     "FALSIFIED",  "#7F1D1D", "#FEE2E2"),
    ("Attention mass = causal necessity",      "FALSIFIED",  "#7F1D1D", "#FEE2E2"),
]

evidence = [
    "§1 r≈1, §3 p=0.00139",
    "§4 full-curve r≈0.05",
    "§5 ~90% trans. share",
    "§5 <25%; top opponents",
    "§4 OOD hijack: 0%",
    "§6 ablation: 0% drop",
    "§6 by elimination",
    "§6 rand ctrl also 0%",
    "§5,§6 independently",
]

y0 = 0.97
row_h = 0.096
for i, ((hyp, verdict, fg, bg), evid) in enumerate(zip(scorecard, evidence)):
    yy = y0 - i * row_h
    ax_h.add_patch(mpatches.FancyBboxPatch(
        (0.0, yy - 0.075), 1.0, row_h - 0.005,
        transform=ax_h.transAxes, fc=bg, ec=C["grid"],
        lw=0.5, boxstyle="round,pad=0.01", clip_on=False,
    ))
    ax_h.text(0.02, yy - 0.025, hyp, transform=ax_h.transAxes,
              fontsize=6.5, va="center", color=C["text"])
    ax_h.text(0.62, yy - 0.025, verdict, transform=ax_h.transAxes,
              fontsize=6.5, va="center", color=fg, fontweight="bold")
    ax_h.text(0.80, yy - 0.025, evid, transform=ax_h.transAxes,
              fontsize=5.8, va="center", color=C["muted"])

# ── 11c. Key Takeaways ──
ax_k = fig.add_subplot(row2[2])
ax_k.set_facecolor(C["panel"])
ax_k.set_title("Key Takeaways", fontsize=9, fontweight="bold",
               color=C["text"], pad=5)
ax_k.axis("off")

takeaways = [
    ("✓", C["s3"], "#D1FAE5",
     "Computation is a trajectory.",
     "Semantic operations unfold as organised geometric expansions across\n"
     "dozens of layers, not at a single point. Conserved across architectures."),

    ("✗", C["s4"], "#FEE2E2",
     "Geometry ≠ Causality.",
     "Geometric maturation and causal power are dissociated. The smooth\n"
     "20-layer trajectory is informationally rich but causally passive."),

    ("✓", C["s5"], "#EDE9FE",
     "MLPs write. Attention routes (and resists).",
     "MLPs are the exclusive builders of the semantic manifold. Attention\n"
     "Heads are the top trajectory opponents, dragging the stream backward."),

    ("✗", C["s6"], "#FEE2E2",
     "Attention weight ≠ causal necessity.",
     "Heads attending 34% to ':' tokens had zero causal impact on ablation.\n"
     "Correlation is not mechanism."),

    ("⚠", C["s4"], "#FEF3C7",
     "Standard controls fail silently.",
     "Degenerate generation, brittle overfits, and generic fragility all\n"
     "masquerade as positive results. Diversity + matched controls are essential."),
]

y0 = 0.97
row_h = 0.185
for i, (icon, col, bg, title, body) in enumerate(takeaways):
    yy = y0 - i * row_h
    ax_k.add_patch(mpatches.FancyBboxPatch(
        (0.0, yy - 0.17), 1.0, row_h - 0.01,
        transform=ax_k.transAxes, fc=bg, ec=C["grid"],
        lw=0.5, boxstyle="round,pad=0.01", clip_on=False,
    ))
    ax_k.add_patch(mpatches.FancyBboxPatch(
        (0.0, yy - 0.17), 0.055, row_h - 0.01,
        transform=ax_k.transAxes, fc=col, ec="none",
        boxstyle="round,pad=0.01", clip_on=False,
    ))
    ax_k.text(0.027, yy - 0.075, icon, transform=ax_k.transAxes,
              fontsize=10, va="center", ha="center", color="white",
              fontweight="bold")
    ax_k.text(0.07, yy - 0.04, title, transform=ax_k.transAxes,
              fontsize=7.2, va="center", color=C["text"], fontweight="bold")
    ax_k.text(0.07, yy - 0.12, body, transform=ax_k.transAxes,
              fontsize=6.2, va="center", color=C["muted"])

# ═══════════════════════════════════════════════════════
# 12.  FOOTER
# ═══════════════════════════════════════════════════════
fig.text(
    0.5, 0.015,
    "Not just what transformers compute — but how they build, store, and (fail to) read computations.",
    ha="center", fontsize=9.5, color=C["muted"], style="italic",
)

# ═══════════════════════════════════════════════════════
# 13.  SAVE
# ═══════════════════════════════════════════════════════
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"Saved -> {OUT.resolve()}")
plt.close(fig)
