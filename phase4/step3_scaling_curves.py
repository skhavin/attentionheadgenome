# -*- coding: utf-8 -*-
# phase4/step3_scaling_curves.py
#
# PURPOSE: Generate the NVIDIA-hook figure:
#   "Theoretical prefill/decode complexity curves WITH vs WITHOUT HeadGenome routing"
#
# OUTPUTS:
#   outputs/phase4/scaling_curves.json   -- data table
#   outputs/phase4/scaling_curves.png    -- publication-quality figure
#
# THEORETICAL MODEL:
#   Baseline (no routing):
#     prefill  = L * H * N^2
#     decode   = L * H * N  (each new token attends to all N cached tokens)
#
#   HeadGenome (per-layer head routing):
#     - f_sink   = fraction of sink heads     → O(1)  per head
#     - f_local  = fraction of local heads    → O(W)  per head, W = window_size
#     - f_ret    = fraction of ret/ind heads  → O(N)  per head (decode: O(N))
#     Effective decode ops = L * (f_sink*1 + f_local*W + f_ret*N) * H_per_layer
#
#   Using empirically measured fractions from cross-architecture results.

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ["HF_HOME"]          = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase4")

# Empirical head fractions from entropy-collapse results
# format: (f_sink, f_local, f_retrieval_induction)
MODEL_FRACTIONS = {
    "GPT-2 Medium\n(MHA, 16H)": {
        "f_sink":  0.039,
        "f_local": 0.810,
        "f_crit":  0.151,  # retrieval + induction
        "color":   "#4C72B0",
    },
    "Qwen-2.5-0.5B\n(GQA-7, 14H)": {
        "f_sink":  0.107,
        "f_local": 0.827,
        "f_crit":  0.066,
        "color":   "#DD8452",
    },
    "Qwen-2.5-1.5B\n(GQA-6, 12H)": {
        "f_sink":  0.012,
        "f_local": 0.878,
        "f_crit":  0.110,
        "color":   "#55A868",
    },
    "Llama-3.2-1B\n(GQA-4, 32H)": {
        "f_sink":  0.000,
        "f_local": 0.850,
        "f_crit":  0.150,
        "color":   "#C44E52",
    },
}

WINDOW_SIZE = 32  # local sliding window
SEQ_LENGTHS = np.array([128, 256, 512, 1024, 2048, 4096, 8192])


def compute_ops(N, fracs, window=WINDOW_SIZE):
    """
    Compute theoretical ops for a single token decode step (relative to baseline).
    Returns (baseline_ops, headgenome_ops) for that N.
    """
    baseline = N   # O(N) per head → normalized to N
    hg = (fracs["f_sink"] * 1 +
          fracs["f_local"] * min(window, N) +
          fracs["f_crit"]  * N)
    return baseline, hg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Compute curves ──────────────────────────────────────────────────
    curves = {}
    for model_name, fracs in MODEL_FRACTIONS.items():
        base_ops, hg_ops = [], []
        savings_pct = []
        for N in SEQ_LENGTHS:
            b, hg = compute_ops(N, fracs)
            base_ops.append(float(b))
            hg_ops.append(float(hg))
            savings_pct.append(float(100 * (1 - hg / b)))
        curves[model_name] = {
            "baseline":     base_ops,
            "headgenome":   hg_ops,
            "savings_pct":  savings_pct,
        }
        safe_name = model_name.replace("\n", " ")
        print(f"\n{safe_name}")
        print(f"  {'N':>6}  {'Baseline':>10}  {'HeadGenome':>12}  {'Savings':>8}")
        for i, N in enumerate(SEQ_LENGTHS):
            print(f"  {N:>6}  {base_ops[i]:>10.1f}  {hg_ops[i]:>12.1f}  {savings_pct[i]:>7.1f}%")

    # ── 2. Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")
    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#CCCCCC")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_color("#444")
        ax.spines["right"].set_color("#444")
        ax.yaxis.label.set_color("#CCCCCC")
        ax.xaxis.label.set_color("#CCCCCC")
        ax.title.set_color("#FFFFFF")

    ax1, ax2 = axes

    # Left: Ops per token at N=4096 (bar chart per model)
    n_target = 4096
    idx_target = list(SEQ_LENGTHS).index(n_target)
    model_labels, base_vals, hg_vals = [], [], []
    for model_name, fracs in MODEL_FRACTIONS.items():
        b, hg = compute_ops(n_target, fracs)
        model_labels.append(model_name.replace("\n", " "))
        base_vals.append(b)
        hg_vals.append(hg)

    x = np.arange(len(model_labels))
    w = 0.35
    bars1 = ax1.bar(x - w/2, base_vals, w, label="Baseline (Full Attn)",
                    color="#4C72B0", alpha=0.85, linewidth=0)
    bars2 = ax1.bar(x + w/2, hg_vals,   w, label="HeadGenome Routing",
                    color="#55A868", alpha=0.85, linewidth=0)

    # Annotate savings
    for xi, (b, hg) in enumerate(zip(base_vals, hg_vals)):
        pct = 100 * (1 - hg / b)
        ax1.text(xi + w/2, hg + 20, f"−{pct:.0f}%",
                 ha="center", va="bottom", fontsize=8, color="#FFD700", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([l.split(" (")[0] for l in model_labels], fontsize=8, color="#CCCCCC")
    ax1.set_ylabel("Decode Ops per Token (normalized)", fontsize=10)
    ax1.set_title(f"Decode Compute at N={n_target}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, facecolor="#2A2D37", edgecolor="#444", labelcolor="#CCC")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", color="#333", linewidth=0.5, linestyle="--")

    # Right: Savings % vs sequence length (line chart)
    for model_name, fracs in MODEL_FRACTIONS.items():
        color = fracs["color"]
        savings_pct = curves[model_name]["savings_pct"]
        label = model_name.replace("\n", " ")
        ax2.plot(SEQ_LENGTHS, savings_pct, color=color, linewidth=2.0, marker="o",
                 markersize=4, label=label)

    ax2.set_xlabel("Sequence Length N", fontsize=10)
    ax2.set_ylabel("Decode FLOP Savings (%)", fontsize=10)
    ax2.set_title("Scaling Curve: Savings vs Sequence Length", fontsize=12, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(SEQ_LENGTHS)
    ax2.set_xticklabels([str(int(n)) for n in SEQ_LENGTHS], fontsize=8, color="#CCCCCC")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=25, color="#666", linewidth=0.8, linestyle="--", label="25% target")
    ax2.axhline(y=40, color="#888", linewidth=0.8, linestyle="--", label="40% target")
    ax2.legend(fontsize=8, facecolor="#2A2D37", edgecolor="#444", labelcolor="#CCC",
               loc="lower right")
    ax2.grid(color="#333", linewidth=0.5, linestyle="--")

    fig.suptitle(
        "HeadGenome Decode-Time Routing: Theoretical FLOP Savings\n"
        "Sink→O(1)  |  Local→O(W=32)  |  Retrieval/Induction→O(N)",
        fontsize=11, color="#FFFFFF", fontweight="bold", y=1.02
    )
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "scaling_curves.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nFigure saved -> {png_path}")

    # ── 3. Save data ───────────────────────────────────────────────────────
    out = {
        "seq_lengths":      [int(n) for n in SEQ_LENGTHS],
        "window_size":      WINDOW_SIZE,
        "model_fractions":  {k: {kk: vv for kk, vv in v.items() if kk != "color"}
                             for k, v in MODEL_FRACTIONS.items()},
        "curves":           {
            k.replace("\n", " "): v for k, v in curves.items()
        },
    }
    json_path = os.path.join(OUT_DIR, "scaling_curves.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Data saved   -> {json_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
