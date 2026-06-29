"""
compute_speedup_curves.py
──────────────────────────
Computes real projected attention-FLOP speedup curves per model using
the empirically measured head-type fractions from canonical_labels.json.

Methodology (same as StreamingLLM, H2O, SnapKV papers):
  Dense attention FLOPs ∝ N²  (quadratic in sequence length)
  Sparse attention FLOPs:
    - Retrieval / Induction heads → keep full causal O(N²)
    - Local heads                 → O(N * W)   where W = window size
    - Sink heads                  → O(N * (sink_size + W))

  Projected speedup = Dense FLOPs / Sparse FLOPs

This is a real, reproducible speedup prediction — not wall-clock on the
torch-mask backend (which adds Python overhead). It matches what you'd
get with a true sparse kernel (FlexAttention, custom CUDA).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW      = 512
SINK_SIZE   = 4
SEQ_LENS    = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])
OUT_DIR     = os.path.join("outputs", "speedup")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS_FILE = os.path.join("outputs", "canonical_labels.json")

MODEL_STYLES = {
    "GPT-2":       {"color": "#4C9EEB", "marker": "o", "full": "GPT-2 Medium (MHA)"},
    "Qwen-0.5B":   {"color": "#F97316", "marker": "s", "full": "Qwen-2.5 0.5B (GQA)"},
    "Llama-3.2-1B":{"color": "#EF4444", "marker": "D", "full": "Llama-3.2-1B (GQA)"},
    "Qwen-1.5B":   {"color": "#A855F7", "marker": "^", "full": "Qwen-2.5 1.5B (GQA)"},
}

# ── Load measured head fractions ───────────────────────────────────────────────
with open(LABELS_FILE) as f:
    raw = json.load(f)

models_data = {}
for name, m in raw["models"].items():
    c = m["counts"]
    total = sum(c.values())
    models_data[name] = {
        "total":     total,
        "f_local":   c.get("local", 0)     / total,
        "f_sink":    c.get("sink", 0)      / total,
        "f_critical": (c.get("retrieval", 0) + c.get("induction", 0)) / total,
        "counts":    c,
        "n_layers":  m["n_layers"],
    }

# ── Compute speedup curves ────────────────────────────────────────────────────
def compute_speedup(f_local, f_sink, f_critical, N, W=WINDOW, sink=SINK_SIZE):
    """
    Returns (prefill_speedup, decode_speedup).

    Prefill: O(N²) → sparse heads reduce to O(N*W)
    Decode:  Each new token attends to KV cache of length N.
             Critical heads: O(N)
             Local heads:    O(W)
             Sink heads:     O(sink + W)
    """
    # Dense ops (normalized)
    dense_prefill = N * N  # ∝ total attention ops in prefill

    # Sparse ops
    sparse_prefill = (f_critical * N * N +
                      f_local    * N * W +
                      f_sink     * N * (sink + W))

    # Decode: each step attends to full KV cache (length ≈ N)
    dense_decode  = N
    sparse_decode = (f_critical * N +
                     f_local    * W +
                     f_sink     * (sink + W))

    prefill_speedup = dense_prefill / sparse_prefill
    decode_speedup  = dense_decode  / sparse_decode
    return prefill_speedup, decode_speedup

results = {}
for name, md in models_data.items():
    prefill_sp, decode_sp = [], []
    for N in SEQ_LENS:
        ps, ds = compute_speedup(md["f_local"], md["f_sink"], md["f_critical"], N)
        prefill_sp.append(ps)
        decode_sp.append(ds)
    results[name] = {
        "prefill": np.array(prefill_sp),
        "decode":  np.array(decode_sp),
    }

# Print summary table
print("\n" + "="*90)
print(f"  {'Model':<18}  {'Type':<8}  {'512':>6} {'1K':>7} {'2K':>7} {'4K':>7} {'8K':>7} {'16K':>7} {'32K':>7}")
print("-"*90)
for name, r in results.items():
    md = models_data[name]
    fc = md["f_critical"]
    fl = md["f_local"]
    fs = md["f_sink"]
    arch = "MHA" if "GPT" in name else "GQA"
    print(f"  {name:<18}  Prefill  " +
          "  ".join(f"{x:6.2f}×" for x in r["prefill"]))
    print(f"  {'':18}  Decode   " +
          "  ".join(f"{x:6.2f}×" for x in r["decode"]))
    print(f"  {'':18}  [{arch}] crit={fc*100:.0f}% local={fl*100:.0f}% sink={fs*100:.0f}%")
    print()
print("="*90)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor="#0F0F0F")
fig.suptitle(
    "HeadGenome: Real Projected Attention-FLOP Speedup\n"
    "Window=512  |  Preserve: Retrieval + Induction heads (full causal)",
    color="white", fontsize=15, fontweight="bold", y=0.98
)

gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35,
              left=0.07, right=0.97, top=0.88, bottom=0.10)

ax_pre   = fig.add_subplot(gs[0, :2])  # Prefill speedup curves
ax_dec   = fig.add_subplot(gs[1, :2])  # Decode speedup curves
ax_bar   = fig.add_subplot(gs[:, 2])   # Bar chart at N=4096

for ax in [ax_pre, ax_dec, ax_bar]:
    ax.set_facecolor("#1A1A2E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color="#222244", linewidth=0.5, alpha=0.8)

x_labels = ["512", "1K", "2K", "4K", "8K", "16K", "32K"]

for name, r in results.items():
    st = MODEL_STYLES.get(name, {"color": "gray", "marker": "o", "full": name})
    ax_pre.plot(x_labels, r["prefill"], color=st["color"], marker=st["marker"],
                linewidth=2.5, markersize=7, label=st["full"])
    ax_dec.plot(x_labels, r["decode"],  color=st["color"], marker=st["marker"],
                linewidth=2.5, markersize=7, label=st["full"])

ax_pre.axhline(1.0, color="#888888", linestyle="--", linewidth=1, alpha=0.6)
ax_dec.axhline(1.0, color="#888888", linestyle="--", linewidth=1, alpha=0.6)

ax_pre.set_title("Prefill Speedup (TTFT)  —  Attention FLOPs: O(N²) → O(N·W)", color="white", fontsize=12)
ax_pre.set_xlabel("Sequence Length (tokens)")
ax_pre.set_ylabel("Speedup ×")
ax_pre.legend(facecolor="#111122", labelcolor="white", fontsize=9, loc="upper left")

ax_dec.set_title("Decode Speedup (TPOT)  —  KV Cache Attention per Step", color="white", fontsize=12)
ax_dec.set_xlabel("Sequence Length (tokens)")
ax_dec.set_ylabel("Speedup ×")
ax_dec.legend(facecolor="#111122", labelcolor="white", fontsize=9, loc="upper left")

# Bar chart at N=4096
N4k_idx = list(SEQ_LENS).index(4096)
bar_names  = [MODEL_STYLES.get(n, {"full": n})["full"].replace(" (MHA)", "\n(MHA)").replace(" (GQA)", "\n(GQA)") for n in results]
bar_pre    = [r["prefill"][N4k_idx] for r in results.values()]
bar_dec    = [r["decode"][N4k_idx]  for r in results.values()]
colors     = [MODEL_STYLES.get(n, {"color": "gray"})["color"] for n in results]

x_pos = np.arange(len(bar_names))
width = 0.35
bars1 = ax_bar.bar(x_pos - width/2, bar_pre, width, label="Prefill", color=colors, alpha=0.9)
bars2 = ax_bar.bar(x_pos + width/2, bar_dec, width, label="Decode",  color=colors, alpha=0.5, hatch="//")

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(bar_names, fontsize=7.5, color="white")
ax_bar.set_title("At N=4096 tokens", color="white", fontsize=12)
ax_bar.set_ylabel("Speedup ×")
ax_bar.axhline(1.0, color="#888888", linestyle="--", linewidth=1, alpha=0.6)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.1f}×",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.1f}×",
                ha="center", va="bottom", fontsize=8, color="#AAAACC")

p1 = mpatches.Patch(facecolor="white", alpha=0.9, label="Prefill (solid)")
p2 = mpatches.Patch(facecolor="white", alpha=0.5, hatch="//", label="Decode (hatched)")
ax_bar.legend(handles=[p1, p2], facecolor="#111122", labelcolor="white", fontsize=9)

# Footnote
fig.text(0.5, 0.01,
    "Methodology: FLOP reduction from measured head-type fractions (canonical_labels.json). "
    "Matches FlexAttention / sparse CUDA kernel projections. "
    "Torch-mask backend adds overhead and is NOT used here.",
    ha="center", color="#888888", fontsize=8)

out_path = os.path.join(OUT_DIR, "figure11_speedup_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
plt.close()
print(f"\n[Done] Figure saved -> {out_path}")

# Save JSON results
json_path = os.path.join(OUT_DIR, "projected_speedup.json")
export = {}
for name, r in results.items():
    md = models_data[name]
    export[name] = {
        "head_fractions": {
            "critical": round(md["f_critical"], 4),
            "local":    round(md["f_local"], 4),
            "sink":     round(md["f_sink"], 4),
        },
        "window": WINDOW,
        "seq_lens": SEQ_LENS.tolist(),
        "prefill_speedup": [round(x, 3) for x in r["prefill"].tolist()],
        "decode_speedup":  [round(x, 3) for x in r["decode"].tolist()],
        "prefill_speedup_at_4k": round(float(r["prefill"][N4k_idx]), 3),
        "decode_speedup_at_4k":  round(float(r["decode"][N4k_idx]),  3),
    }
with open(json_path, "w") as f:
    json.dump(export, f, indent=2)
print(f"[Done] JSON  saved -> {json_path}")
