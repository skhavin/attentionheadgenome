import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load causal results
    with open("../outputs/causal_intervention/sweep_results.json", "r") as f:
        results = json.load(f)
        
    with open("../outputs/intra_mapping/f_statistic_data.json", "r") as f:
        f_ratios = json.load(f)
        
    qwen_f = f_ratios["Qwen2.5-1.5B"]["real_F"]
    layers = list(range(len(qwen_f)))
    
    # Process hijack rates
    # We take the max Real_B where p < 0.05. If no c has p < 0.05, we take 0.
    def get_hijack_curve(pair_key):
        curve = []
        for l in layers:
            l_str = str(l)
            best_rate = 0
            if l_str in results[pair_key]:
                for c in ["1.5", "3.0", "5.0"]:
                    if c in results[pair_key][l_str]:
                        cell = results[pair_key][l_str][c]
                        if cell["mcnemar_p"] < 0.05:
                            best_rate = max(best_rate, cell["b_rate_real"])
            curve.append(best_rate)
        return curve

    curve_arith = get_hijack_curve("arithmetic_to_sorting")
    curve_fact = get_hijack_curve("fact_recall_to_comparison")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('F-Statistic (Manifold Maturation)', color=color)
    ax1.plot(layers, qwen_f, color=color, linewidth=2, label='F-Ratio')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('Target Hijack Success Rate (%)', color='black')
    ax2.plot(layers, np.array(curve_arith) * 100, color=color2, linestyle='--', marker='o', label='Arith->Sort')
    ax2.plot(layers, np.array(curve_fact) * 100, color=color3, linestyle='--', marker='s', label='Fact->Comp')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(-5, 100)

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.9))
    plt.title("Causal Hijack Success vs. Geometric Maturation (Qwen2.5-1.5B)")
    plt.grid(True, alpha=0.3)
    plt.savefig("../outputs/causal_intervention/causal_vs_fratio.png", dpi=300)
    print("Plot saved to outputs/causal_intervention/causal_vs_fratio.png")

    # Compute correlation
    peak_l = np.argmax(qwen_f)
    print(f"F-Ratio Peak Layer: {peak_l}")
    
    from scipy.stats import pearsonr
    r_arith_full, _ = pearsonr(qwen_f, curve_arith)
    r_fact_full, _ = pearsonr(qwen_f, curve_fact)
    
    print(f"Full curve correlation Arith: {r_arith_full:.3f}")
    print(f"Full curve correlation Fact: {r_fact_full:.3f}")

if __name__ == "__main__":
    main()
