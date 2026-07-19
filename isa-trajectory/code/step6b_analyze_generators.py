import json
import numpy as np

def main():
    with open("../outputs/generator_analysis/dta_results_fact.json", "r") as f:
        dta = json.load(f)
        
    static_heads = np.array(dta["static_heads"]) # [num_layers, num_heads]
    static_mlps = np.array(dta["static_mlps"])   # [num_layers]
    trans_heads = np.array(dta["trans_heads"])
    trans_mlps = np.array(dta["trans_mlps"])
    
    num_layers, num_heads = static_heads.shape
    
    print("\n--- Top Trajectory Generators (Static DTA) ---")
    flat_static = []
    for l in range(num_layers):
        flat_static.append((f"L{l} MLP", static_mlps[l], trans_mlps[l]))
        for h in range(num_heads):
            flat_static.append((f"L{l} H{h}", static_heads[l, h], trans_heads[l, h]))
            
    flat_static.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i in range(15):
        name, stat, trans = flat_static[i]
        flag = " [! NORM ARTIFACT]" if stat > 0 and trans < 0.05 else "" # Flag discrepancy
        print(f"{i+1}. {name:10s} | Static: {stat:6.3f} | Transition: {trans*100:6.2f}% {flag}")

    print("\n--- Top Trajectory Generators (Transition DTA, Rise Phase L10-20) ---")
    flat_trans = []
    for l in range(num_layers):
        if 10 <= l <= 20:
            flat_trans.append((f"L{l} MLP", static_mlps[l], trans_mlps[l]))
            for h in range(num_heads):
                flat_trans.append((f"L{l} H{h}", static_heads[l, h], trans_heads[l, h]))
                
    flat_trans.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i in range(15):
        name, stat, trans = flat_trans[i]
        print(f"{i+1}. {name:10s} | Transition: {trans*100:6.2f}% | Static: {stat:6.3f}")

    print("\n--- Top Trajectory OPPONENTS (Negative Transition DTA, Rise Phase L10-20) ---")
    flat_trans_opp = sorted(flat_trans, key=lambda x: x[2]) # Most negative first
    for i in range(10):
        name, stat, trans = flat_trans_opp[i]
        print(f"{i+1}. {name:10s} | Transition: {trans*100:6.2f}% | Static: {stat:6.3f}")

if __name__ == "__main__":
    main()
