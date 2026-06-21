# Generate Figure 5: PPL vs budget on LONG concatenated WikiText documents.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import PHASE3_DIR, KV_BUDGETS

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"ours": "#2dd4bf", "sllm": "#94a3b8", "h2o": "#f87171", "full": "#a78bfa"}

with open(os.path.join(PHASE3_DIR, "benchmark_long_results.pkl"), "rb") as f:
    results = pickle.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
full_ppl = results["full"]["ppl"]

for label, prefix, color, marker in [
    ("Proactive (ours)", "proactive", COLORS["ours"], "o"),
    ("StreamingLLM", "streamingllm", COLORS["sllm"], "s"),
    ("H2O", "h2o", COLORS["h2o"], "^"),
]:
    ppls, budgets_used = [], []
    for b in KV_BUDGETS:
        key = f"{prefix}_{b}"
        if key in results:
            ppls.append(results[key]["ppl"])
            budgets_used.append(b)
    ax.plot(budgets_used, ppls, marker=marker, label=label, color=color, linewidth=2.5, markersize=8, zorder=3)

ax.axhline(y=full_ppl, color=COLORS["full"], linestyle="--", linewidth=1.5,
           label=f"Full Attention (PPL={full_ppl:.1f})", alpha=0.7)

ax.set_xlabel("KV Cache Budget (tokens retained)")
ax.set_ylabel("Perplexity (lower is better)")
ax.set_title("Perplexity vs KV Cache Budget — Long Documents (1024 tokens)")
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.set_xticks(KV_BUDGETS)
ax.grid(True, alpha=0.2)

path = os.path.join(FIG_DIR, "fig5_ppl_long_docs.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

# Print derived metrics
eval_tokens = 100
num_docs = 6
avg_seq = 1024
print(f"\n{'Method':<20} {'Budget':>6} {'PPL':>8} {'Deg%':>8} {'VRAM':>8} {'Tok/s':>8} {'Comp%':>8}")
print("-" * 78)
for key in ["full"] + [f"{m}_{b}" for b in KV_BUDGETS for m in ["proactive", "streamingllm", "h2o"]]:
    if key not in results:
        continue
    r = results[key]
    budget = int(key.split("_")[-1]) if "_" in key else avg_seq
    method = key.split("_")[0] if "_" in key else "Full"
    deg = ((r["ppl"] - full_ppl) / full_ppl) * 100
    tok_s = (eval_tokens * num_docs) / r["time_s"]
    comp = (budget / avg_seq) * 100
    print(f"{method:<20} {budget:>6} {r['ppl']:>8.2f} {deg:>7.1f}% {r['vram_mb']:>7.0f} {tok_s:>7.1f} {comp:>7.1f}%")
