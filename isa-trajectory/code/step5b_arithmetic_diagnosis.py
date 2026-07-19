import json
import torch
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import binomtest

def classify_arithmetic_to_sorting(text):
    if len(re.findall(r',', text)) >= 2 or len(re.findall(r'\n\d+\.', text)) >= 2 or len(re.findall(r'\n-', text)) >= 2:
        return 'B' 
    if re.search(r'\d+', text):
        return 'A' 
    return 'C' 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    def get_steering_vector(src_cat, tgt_cat, layer):
        src_idx = cat_indices[src_cat]
        tgt_idx = cat_indices[tgt_cat]
        src_centroid = raw_T[src_idx].mean(dim=0).to(model.dtype)
        tgt_centroid = raw_T[tgt_idx].mean(dim=0).to(model.dtype)
        return (tgt_centroid - src_centroid)[layer].to(model.device)

    # Falsifiable Test: Does the presence of '\n' break the hijack?
    # Let's generate 20 novel prompts without the '\n' (matching original structure)
    # e.g., "What does 61 + 24 equal? Answer:"
    print("\n--- Falsifiable Test: Reintroducing Original Single-Line Format ---")
    fixed_prompts = []
    np.random.seed(42)
    for _ in range(20):
        x = np.random.randint(10, 99)
        y = np.random.randint(10, 99)
        fixed_prompts.append(f"What does {x} + {y} equal? Answer:")
        
    layer = 27
    c = 1.5
    v_steer = get_steering_vector("arithmetic", "sorting", layer)
    target_module = model.model.layers[layer]
    
    inputs = tokenizer(fixed_prompts, return_tensors="pt", padding=True).to(model.device)
    
    def hook_real(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] += c * v_steer
            return (h,) + output[1:]
        else:
            output[:, -1, :] += c * v_steer
            return output
            
    handle = target_module.register_forward_hook(hook_real)
    with torch.no_grad():
        gen_real = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    handle.remove()
    
    outputs_real = tokenizer.batch_decode(gen_real[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    buckets_real = [classify_arithmetic_to_sorting(txt) for txt in outputs_real]
    b_rate = buckets_real.count('B') / len(buckets_real)
    print(f"Hijack Rate with Fixed Format (Single-Line): {b_rate*100:.1f}%")

if __name__ == "__main__":
    main()
