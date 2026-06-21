# Combine benchmark and baseline results into a formatted table.
# Output: printed table + saved as results_table.txt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
from config import PHASE3_DIR, KV_BUDGETS

def main():
    # Load results
    bench_path = os.path.join(PHASE3_DIR, "benchmark_results.pkl")
    base_path = os.path.join(PHASE3_DIR, "baseline_results.pkl")

    with open(bench_path, "rb") as f:
        bench = pickle.load(f)
    with open(base_path, "rb") as f:
        base = pickle.load(f)

    # Merge
    all_results = {**bench, **base}

    # Print table
    header = f"{'Method':<20} {'Budget':>6} {'PPL':>8} {'VRAM(MB)':>9} {'Time(s)':>8}"
    sep = "-" * len(header)
    lines = [header, sep]

    # Full attention first
    if "full" in all_results:
        r = all_results["full"]
        lines.append(f"{'Full Attention':<20} {'all':>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")

    lines.append(sep)

    for budget in KV_BUDGETS:
        for method, key in [("StreamingLLM", f"streamingllm_{budget}"),
                            ("H2O", f"h2o_{budget}"),
                            ("Proactive (ours)", f"proactive_{budget}")]:
            if key in all_results:
                r = all_results[key]
                lines.append(f"{method:<20} {budget:>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")
        lines.append(sep)

    table = "\n".join(lines)
    print(table)

    save_path = os.path.join(PHASE3_DIR, "results_table.txt")
    with open(save_path, "w") as f:
        f.write(table)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()
