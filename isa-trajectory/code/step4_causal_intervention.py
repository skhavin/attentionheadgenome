import json
import torch
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import binomtest

# 1. Bucketing Classifiers
def classify_arithmetic_to_sorting(text):
    # Target (Bucket B): Commas or newlines indicating a list
    if len(re.findall(r',', text)) >= 2 or len(re.findall(r'\n\d+\.', text)) >= 2 or len(re.findall(r'\n-', text)) >= 2:
        return 'B' # Target (Hijacked)
    
    # Source (Bucket A): Contains a number (but isn't a list)
    if re.search(r'\d+', text):
        return 'A' # Source (Correct)
        
    return 'C' # Garbage/Neither

def classify_fact_to_comparison(text):
    # Target (Bucket B): Comparative terms
    comparatives = ['yes', 'no', 'true', 'false', 'smaller', 'larger', 'taller', 'older', 'bigger', 'greater', 'less', 'better', 'worse', 'higher', 'lower']
    text_lower = text.lower()
    
    tokens = re.findall(r'\b\w+\b', text_lower)
    if any(c in tokens for c in comparatives):
        return 'B' # Target (Hijacked)
        
    # Source (Bucket A): If it outputs non-comparative words (likely entity names)
    if len(tokens) > 0:
        return 'A' # Source (Correct)
        
    return 'C' # Garbage/Neither

def mcnemar_exact(n12, n21):
    n = n12 + n21
    if n == 0: return 1.0
    k = min(n12, n21)
    return binomtest(k, n, p=0.5).pvalue

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    print("Loading Data...")
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
        
    pairs = [
        ("arithmetic", "sorting", classify_arithmetic_to_sorting),
        ("fact_recall", "comparison", classify_fact_to_comparison)
    ]
    
    n_layers = model.config.num_hidden_layers
    c_values = [1.5, 3.0, 5.0]
    results = {}
    
    for src_cat, tgt_cat, classifier in pairs:
        print(f"\n{'='*50}\nSweep: {src_cat} -> {tgt_cat}\n{'='*50}")
        src_idx = cat_indices[src_cat]
        tgt_idx = cat_indices[tgt_cat]
        
        src_centroid = raw_T[src_idx].mean(dim=0).to(model.dtype)
        tgt_centroid = raw_T[tgt_idx].mean(dim=0).to(model.dtype)
        v_steer_all_layers = (tgt_centroid - src_centroid)
        
        src_texts = cat_prompts[src_cat]
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True).to(model.device)
        
        results[f"{src_cat}_to_{tgt_cat}"] = {}
        
        for layer in range(n_layers):
            results[f"{src_cat}_to_{tgt_cat}"][layer] = {}
            v_steer = v_steer_all_layers[layer].to(model.device)
            v_rand = torch.randn_like(v_steer)
            v_rand = v_rand / torch.norm(v_rand) * torch.norm(v_steer)
            
            for c in c_values:
                def hook_real(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h[:, -1, :] += c * v_steer
                        return (h,) + output[1:]
                    else:
                        output[:, -1, :] += c * v_steer
                        return output
                    
                def hook_rand(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h[:, -1, :] += c * v_rand
                        return (h,) + output[1:]
                    else:
                        output[:, -1, :] += c * v_rand
                        return output
                
                target_module = model.model.layers[layer]
                
                # Real Vector
                handle = target_module.register_forward_hook(hook_real)
                with torch.no_grad():
                    gen_real = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                handle.remove()
                
                # Random Vector
                handle = target_module.register_forward_hook(hook_rand)
                with torch.no_grad():
                    gen_rand = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                handle.remove()
                
                # Baseline (no intervention)
                if c == c_values[0]: # only run baseline once per layer
                    with torch.no_grad():
                        gen_base = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                    outputs_base = tokenizer.batch_decode(gen_base[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    buckets_base = [classifier(txt) for txt in outputs_base]
                    results[f"{src_cat}_to_{tgt_cat}"][layer]["baseline_B"] = buckets_base.count('B') / len(buckets_base)
                
                outputs_real = tokenizer.batch_decode(gen_real[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                outputs_rand = tokenizer.batch_decode(gen_rand[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                buckets_real = [classifier(txt) for txt in outputs_real]
                buckets_rand = [classifier(txt) for txt in outputs_rand]
                
                b_rate_real = buckets_real.count('B') / len(buckets_real)
                b_rate_rand = buckets_rand.count('B') / len(buckets_rand)
                
                n12 = n21 = 0
                for r_real, r_rand in zip(buckets_real, buckets_rand):
                    if r_real == 'B' and r_rand != 'B': n12 += 1
                    elif r_real != 'B' and r_rand == 'B': n21 += 1
                
                p_val = mcnemar_exact(n12, n21)
                
                results[f"{src_cat}_to_{tgt_cat}"][layer][c] = {
                    "real_A": buckets_real.count('A'),
                    "real_B": buckets_real.count('B'),
                    "real_C": buckets_real.count('C'),
                    "rand_A": buckets_rand.count('A'),
                    "rand_B": buckets_rand.count('B'),
                    "rand_C": buckets_rand.count('C'),
                    "b_rate_real": b_rate_real,
                    "b_rate_rand": b_rate_rand,
                    "mcnemar_p": float(p_val)
                }
                print(f"[L{layer:02d}, c={c:.1f}] Real B: {b_rate_real:.2f}, Rand B: {b_rate_rand:.2f} | p={p_val:.4f}")

    os.makedirs("../outputs/causal_intervention", exist_ok=True)
    with open("../outputs/causal_intervention/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
