import json
import os
import matplotlib.pyplot as plt

def plot_r2():
    models = ["Qwen2.5-1.5B", "Llama-3.2-1B", "phi-1_5"]
    out_dir = "../outputs/trajectories"
    
    plt.figure(figsize=(10, 6))
    
    for m in models:
        path = os.path.join(out_dir, m, "r2_values.json")
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
            
        with open(path, "r") as f:
            r2 = json.load(f)
            
        # Normalize x-axis to be percentage of depth so models can be compared
        # or just plot absolute depth
        plt.plot(range(len(r2)), r2, marker='o', label=f"{m} (L={len(r2)})")
        
    plt.title("Confound Regression R² vs. Layer Depth")
    plt.xlabel("Layer Index")
    plt.ylabel("R² (Proportion of Variance Explained)")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../../r2_vs_layer.png", dpi=300)
    print("Saved plot to r2_vs_layer.png")

if __name__ == "__main__":
    plot_r2()
