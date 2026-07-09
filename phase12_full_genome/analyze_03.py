import json
import numpy as np

def analyze_json():
    with open('03_qk_ov_ablation_results.json', 'r') as f:
        data = json.load(f)
    
    raw = data.get("raw_results", {})
    cond_a = raw.get("condition_A_q_permute", {})
    cond_b = raw.get("condition_B_zero_ov", {})
    
    print("# Gate A Deep Data Analysis (Script 03)")
    print("## Delta PPL Distributions by Canonical Class\n")
    
    for arch in cond_b.keys():
        print(f"### {arch}")
        a_data = cond_a.get(arch, [])
        b_data = cond_b.get(arch, [])
        
        # Group by label
        a_by_label = {}
        for r in a_data:
            lbl = r.get("canonical_label", "unknown")
            a_by_label.setdefault(lbl, []).append(r["delta_ppl"])
            
        b_by_label = {}
        for r in b_data:
            lbl = r.get("canonical_label", "unknown")
            b_by_label.setdefault(lbl, []).append(r["delta_ppl"])
            
        # Print Condition A
        print("**Condition A (Q Permute) - Delta PPL:**")
        for lbl in sorted(a_by_label.keys()):
            vals = [v for v in a_by_label[lbl] if np.isfinite(v)]
            if not vals: continue
            print(f"- {lbl.capitalize()} (n={len(vals)}): Mean = {np.mean(vals):.4f}, Std = {np.std(vals):.4f}, Max = {np.max(vals):.4f}")
            
        # Print Condition B
        print("\n**Condition B (Zero OV) - Delta PPL:**")
        for lbl in sorted(b_by_label.keys()):
            vals = [v for v in b_by_label[lbl] if np.isfinite(v)]
            if not vals: continue
            print(f"- {lbl.capitalize()} (n={len(vals)}): Mean = {np.mean(vals):.4f}, Std = {np.std(vals):.4f}, Max = {np.max(vals):.4f}")
        
        print("\n---\n")

if __name__ == "__main__":
    analyze_json()
