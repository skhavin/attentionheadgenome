import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quadratic_func(x, a, b, c):
    return a * (x**2) + b * x + c

# Load the empirical JSON data
print("Loading TTFT results...")
with open("ttft_results.json", "r") as f:
    results = json.load(f)

baseline = results["baseline"]
router = results["router_simulated"]

seq_lengths = sorted([int(k) for k in baseline.keys()])
base_times = [baseline[str(k)] for k in seq_lengths]
router_times = [router[str(k)] for k in seq_lengths]

N_vals = np.array(seq_lengths)
B_vals = np.array(base_times)
R_vals = np.array(router_times)

# Fit curves to mathematically extract the O(N^2) coefficient
popt_base, _ = curve_fit(quadratic_func, N_vals, B_vals)
popt_router, _ = curve_fit(quadratic_func, N_vals, R_vals)

a_base, b_base, c_base = popt_base
a_router, b_router, c_router = popt_router

print(f"Mathematical Complexity Extraction:")
print(f"  Baseline O(N^2) Coefficient: {a_base:.2e}")
print(f"  Router O(N^2) Coefficient:   {a_router:.2e}")
print(f"  Quadratic Suppression Factor: {a_base / a_router:.1f}x")

# Plotting - Academic Light Mode Theme
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('default')

# Plot the empirical scatter points
plt.scatter(N_vals, B_vals, color='#ef4444', label='Baseline Dense (Empirical)', s=60, zorder=5)
plt.scatter(N_vals, R_vals, color='#3b82f6', label='Hybrid Dense Router (Simulated GPU Bound)', s=60, zorder=5)

# Plot the theoretical fitted curves
x_smooth = np.linspace(min(N_vals), max(N_vals), 500)
plt.plot(x_smooth, quadratic_func(x_smooth, *popt_base), color='#ef4444', linestyle='--', alpha=0.7, 
         label=f'Fit: $O(N^2)$ (a={a_base:.1e})')
plt.plot(x_smooth, quadratic_func(x_smooth, *popt_router), color='#3b82f6', linestyle='--', alpha=0.7, 
         label=f'Fit: $O(N)$ dominated (a={a_router:.1e})')

plt.title('Time To First Token (TTFT) Scaling vs Sequence Length\n(Qwen2.5-0.5B Prefill Phase)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sequence Length (Tokens)', fontsize=12, fontweight='bold')
plt.ylabel('TTFT Latency (Milliseconds)', fontsize=12, fontweight='bold')

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='black')

# Annotate the polynomial suppression
plt.annotate(f"{a_base/a_router:.1f}x $\mathcal{{O}}(N^2)$ Suppression", 
             xy=(x_smooth[-1], quadratic_func(x_smooth[-1], *popt_router)),
             xytext=(x_smooth[-1]*0.6, quadratic_func(x_smooth[-1], *popt_base)*0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("figure14_ttft_speedup.png", bbox_inches='tight')
print("\nPlot saved as figure14_ttft_speedup.png")
