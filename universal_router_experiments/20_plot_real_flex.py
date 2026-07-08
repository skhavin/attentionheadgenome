import json
import matplotlib.pyplot as plt
import numpy as np

# Load real C++ FlexAttention hardware results
with open("flex_ttft_results.json", "r") as f:
    results = json.load(f)

seq_lens = [int(k) for k in results["baseline_sdpa"].keys()]
dense_times = [results["baseline_sdpa"][str(k)] for k in seq_lens]
flex_times = [results["compiled_flex_hybrid"][str(k)] for k in seq_lens]

# Setup plot styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Plot raw hardware timings
ax.plot(seq_lens, dense_times, color='#ff7b72', marker='o', linewidth=2.5, markersize=6, label='Baseline (Dense SDPA C++)')
ax.plot(seq_lens, flex_times, color='#79c0ff', marker='s', linewidth=2.5, markersize=6, label='Hybrid Router (FlexAttention C++)')

# Trend fitting (Polynomial degree 2 for quadratic scaling bounds)
x_fit = np.linspace(min(seq_lens), max(seq_lens), 100)
z_dense = np.polyfit(seq_lens, dense_times, 2)
z_flex = np.polyfit(seq_lens, flex_times, 2)
p_dense = np.poly1d(z_dense)
p_flex = np.poly1d(z_flex)

ax.plot(x_fit, p_dense(x_fit), color='#ff7b72', linestyle='--', alpha=0.5)
ax.plot(x_fit, p_flex(x_fit), color='#79c0ff', linestyle='--', alpha=0.5)

# Styling and grid
ax.set_title('Real Hardware TTFT: Native C++ FlexAttention vs SDPA', color='white', fontsize=14, pad=20, weight='bold')
ax.set_xlabel('Context Sequence Length (N)', color='#c9d1d9', fontsize=12, labelpad=10)
ax.set_ylabel('Execution Time (ms) - Lower is Better', color='#c9d1d9', fontsize=12, labelpad=10)
ax.grid(color='#30363d', linestyle='--', linewidth=0.5, alpha=0.7)
ax.tick_params(colors='#c9d1d9', which='both', labelsize=10)

for spine in ax.spines.values():
    spine.set_color('#30363d')
    spine.set_linewidth(1.5)

legend = ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11, loc='upper left')
for text in legend.get_texts():
    text.set_color('white')

plt.tight_layout()
plt.savefig('../figure15_real_hardware_speedup.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
print("Saved real hardware speedup plot.")
