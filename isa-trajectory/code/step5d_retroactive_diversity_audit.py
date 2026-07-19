import json
import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import binomtest

def classify_fact_to_comparison(text):
    comparatives = ['yes', 'no', 'true', 'false', 'smaller', 'larger', 'taller', 'older', 'bigger', 'greater', 'less', 'better', 'worse', 'higher', 'lower']
    text_lower = text.lower()
    tokens = re.findall(r'\b\w+\b', text_lower)
    if any(c in tokens for c in comparatives):
        return 'B' 
    if len(tokens) > 0:
        return 'A' 
    return 'C' 

def classify_arithmetic_to_sorting(text):
    if len(re.findall(r',', text)) >= 2 or len(re.findall(r'\n\d+\.', text)) >= 2 or len(re.findall(r'\n-', text)) >= 2:
        return 'B' 
    if re.search(r'\d+', text):
        return 'A' 
    return 'C' 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    with open("../outputs/causal_intervention/sweep_results.json", "r") as f:
        sweep_data = json.load(f)
        
    significant_cells = []
    
    for pair_key, pair_data in sweep_data.items():
        if pair_key == "arithmetic_to_sorting":
            src_cat, tgt_cat = "arithmetic", "sorting"
            classifier = classify_arithmetic_to_sorting
        else:
            src_cat, tgt_cat = "fact_recall", "comparison"
            classifier = classify_fact_to_comparison
            
        for layer_str, layer_data in pair_data.items():
            if layer_str == "baseline_B": continue
            layer = int(layer_str)
            for c_str, metrics in layer_data.items():
                if c_str == "baseline_B": continue
                c = float(c_str)
                if metrics["mcnemar_p"] < 0.05 and metrics["b_rate_real"] >= 0.30:
                    significant_cells.append({
                        "src_cat": src_cat, "tgt_cat": tgt_cat,
                        "layer": layer, "c": c, "p": metrics["mcnemar_p"], "b_rate": metrics["b_rate_real"],
                        "classifier": classifier
                    })
                    
    print(f"Found {len(significant_cells)} significant cells in the original sweep.")
    if len(significant_cells) == 0:
        print("No significant cells found. Exiting.")
        return
        
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        val_prompts = json.load(f)
        
    categories_all = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    cat_indices = {c: [] for c in categories_all}
    for i, p in enumerate(val_prompts):
        cat_indices[p["task_type"]].append(i)

    raw_T = torch.load("../outputs/trajectories/Qwen2.5-1.5B/val_raw_trajectories.pt", map_location="cpu")
    
    def get_steering_vector(src, tgt, l):
        src_idx = cat_indices[src]
        tgt_idx = cat_indices[tgt]
        src_centroid = raw_T[src_idx].mean(dim=0).to(model.dtype)
        tgt_centroid = raw_T[tgt_idx].mean(dim=0).to(model.dtype)
        return (tgt_centroid - src_centroid)[l].to(model.device)

    genuine_gates = []
    
    for cell in significant_cells:
        print(f"\nAuditing: {cell['src_cat']}->{cell['tgt_cat']} | Layer {cell['layer']} | c={cell['c']} | Orig B: {cell['b_rate']*100:.1f}%")
        
        # We run it on the EXACT SAME 30 validation prompts that produced the significance
        prompts = [val_prompts[i]["prompt"] for i in cat_indices[cell['src_cat']]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        v_steer = get_steering_vector(cell["src_cat"], cell["tgt_cat"], cell["layer"])
        target_module = model.model.layers[cell["layer"]]
        
        def hook_real(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h[:, -1, :] += cell["c"] * v_steer
                return (h,) + output[1:]
            else:
                output[:, -1, :] += cell["c"] * v_steer
                return output
                
        handle = target_module.register_forward_hook(hook_real)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        handle.remove()
        
        outputs = tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Compute uniqueness fraction
        # Only look at unique outputs for prompts that actually HIT Bucket B (the hijack)
        hijacked_outputs = []
        for txt in outputs:
            if cell["classifier"](txt) == 'B':
                hijacked_outputs.append(txt.strip())
                
        if len(hijacked_outputs) == 0:
            print("  Warning: No hijacked outputs found in re-run. Skipping diversity check.")
            continue
            
        unique_hijacks = set(hijacked_outputs)
        fraction_unique = len(unique_hijacks) / len(hijacked_outputs)
        
        print(f"  Total Hijacked Outputs: {len(hijacked_outputs)}")
        print(f"  Unique Strings: {len(unique_hijacks)}")
        print(f"  Fraction Unique: {fraction_unique*100:.1f}%")
        
        if fraction_unique > 0.4:
            print(f"  >>> PASSED DIVERSITY BAR <<<")
            genuine_gates.append(cell)
            print("  Sample outputs:")
            for out in list(unique_hijacks)[:5]:
                print(f"    '{out}'")
        else:
            print(f"  >>> FAILED DIVERSITY BAR (Degenerate) <<<")
            print("  Sample outputs:")
            for out in list(unique_hijacks)[:2]:
                print(f"    '{out}'")
                
    print(f"\n=========================================")
    print(f"Retroactive Audit Complete.")
    print(f"Total Significant Cells: {len(significant_cells)}")
    print(f"Genuine Gates (High Diversity): {len(genuine_gates)}")
    print(f"=========================================")

if __name__ == "__main__":
    main()
