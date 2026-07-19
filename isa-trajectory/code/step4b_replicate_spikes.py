import json
import torch
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import binomtest

# 1. Bucketing Classifiers
def classify_arithmetic_to_sorting(text):
    if len(re.findall(r',', text)) >= 2 or len(re.findall(r'\n\d+\.', text)) >= 2 or len(re.findall(r'\n-', text)) >= 2:
        return 'B' 
    if re.search(r'\d+', text):
        return 'A' 
    return 'C' 

def classify_fact_to_comparison(text):
    comparatives = ['yes', 'no', 'true', 'false', 'smaller', 'larger', 'taller', 'older', 'bigger', 'greater', 'less', 'better', 'worse', 'higher', 'lower']
    text_lower = text.lower()
    tokens = re.findall(r'\b\w+\b', text_lower)
    if any(c in tokens for c in comparatives):
        return 'B' 
    if len(tokens) > 0:
        return 'A' 
    return 'C' 

def mcnemar_exact(n12, n21):
    n = n12 + n21
    if n == 0: return 1.0
    k = min(n12, n21)
    return binomtest(k, n, p=0.5).pvalue

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        val_prompts = json.load(f)
        
    cat_prompts = {c: [] for c in ["arithmetic", "sorting", "fact_recall", "comparison"]}
    for p in val_prompts:
        if p["task_type"] in cat_prompts:
            cat_prompts[p["task_type"]].append(p["prompt"])
            
    raw_T = torch.load("../outputs/trajectories/Qwen2.5-1.5B/val_raw_trajectories.pt", map_location="cpu")
    categories_all = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    cat_indices = {c: [] for c in categories_all}
    for i, p in enumerate(val_prompts):
        cat_indices[p["task_type"]].append(i)

    # The top cells to replicate
    targets = [
        {"src": "arithmetic", "tgt": "sorting", "clf": classify_arithmetic_to_sorting, "layer": 27, "c": 1.5, "orig_real": 0.43},
        {"src": "fact_recall", "tgt": "comparison", "clf": classify_fact_to_comparison, "layer": 11, "c": 5.0, "orig_real": 0.83},
        {"src": "fact_recall", "tgt": "comparison", "clf": classify_fact_to_comparison, "layer": 25, "c": 3.0, "orig_real": 0.50},
        {"src": "fact_recall", "tgt": "comparison", "clf": classify_fact_to_comparison, "layer": 27, "c": 3.0, "orig_real": 0.73},
    ]

    seeds = [42, 100, 200, 300]
    
    results = []

    for t in targets:
        src_cat, tgt_cat, classifier, layer, c = t["src"], t["tgt"], t["clf"], t["layer"], t["c"]
        print(f"\nReplicating {src_cat}->{tgt_cat} Layer {layer} c={c} (Orig Real B: {t['orig_real']})")
        
        src_idx = cat_indices[src_cat]
        tgt_idx = cat_indices[tgt_cat]
        
        src_centroid = raw_T[src_idx].mean(dim=0).to(model.dtype)
        tgt_centroid = raw_T[tgt_idx].mean(dim=0).to(model.dtype)
        v_steer = (tgt_centroid - src_centroid)[layer].to(model.device)
        
        inputs = tokenizer(cat_prompts[src_cat], return_tensors="pt", padding=True).to(model.device)
        
        # Real Vector (Deterministic since do_sample=False)
        def hook_real(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h[:, -1, :] += c * v_steer
                return (h,) + output[1:]
            else:
                output[:, -1, :] += c * v_steer
                return output
                
        target_module = model.model.layers[layer]
        handle = target_module.register_forward_hook(hook_real)
        with torch.no_grad():
            gen_real = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        handle.remove()
        
        outputs_real = tokenizer.batch_decode(gen_real[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        buckets_real = [classifier(txt) for txt in outputs_real]
        b_rate_real = buckets_real.count('B') / len(buckets_real)
        print(f"Real B Rate (Deterministic): {b_rate_real:.2f}")

        # Random Vectors
        rand_b_rates = []
        p_values = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            v_rand = torch.randn_like(v_steer)
            v_rand = v_rand / torch.norm(v_rand) * torch.norm(v_steer)
            
            def hook_rand(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h[:, -1, :] += c * v_rand
                    return (h,) + output[1:]
                else:
                    output[:, -1, :] += c * v_rand
                    return output
                    
            handle = target_module.register_forward_hook(hook_rand)
            with torch.no_grad():
                gen_rand = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            handle.remove()
            
            outputs_rand = tokenizer.batch_decode(gen_rand[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            buckets_rand = [classifier(txt) for txt in outputs_rand]
            
            b_rate_rand = buckets_rand.count('B') / len(buckets_rand)
            rand_b_rates.append(b_rate_rand)
            
            n12 = n21 = 0
            for r_real, r_rand in zip(buckets_real, buckets_rand):
                if r_real == 'B' and r_rand != 'B': n12 += 1
                elif r_real != 'B' and r_rand == 'B': n21 += 1
            
            p_val = mcnemar_exact(n12, n21)
            p_values.append(p_val)
            print(f"  Seed {seed} | Rand B: {b_rate_rand:.2f} | p={p_val:.4f}")
            
        results.append({
            "pair": f"{src_cat}->{tgt_cat}",
            "layer": layer,
            "c": c,
            "real_b": b_rate_real,
            "rand_b_seeds": rand_b_rates,
            "p_values": p_values
        })
        
    with open("../outputs/causal_intervention/robustness_check.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
