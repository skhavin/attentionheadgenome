import json
import torch
import numpy as np
import os
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

def mcnemar_exact(n12, n21):
    n = n12 + n21
    if n == 0: return 1.0
    k = min(n12, n21)
    return binomtest(k, n, p=0.5).pvalue

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

    countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua", "Argentina", "Armenia", "Australia", "Austria",
        "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
        "Bolivia", "Bosnia", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
        "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia", "Comoros", "Costa Rica", "Croatia", "Cuba",
        "Cyprus", "Denmark", "Djibouti", "Dominica", "Ecuador", "Egypt", "El Salvador", "Eritrea", "Estonia", "Eswatini",
        "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece",
        "Grenada", "Guatemala", "Guinea", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
        "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya"
    ]
    
    fact_batches = []
    all_prompts = []
    for i in range(4):
        batch = [f"The capital of {c} is\nAnswer:" for c in countries[i*20:(i+1)*20]]
        fact_batches.append(batch)
        all_prompts.extend(batch)
        
    layer = 11
    c = 5.0
    v_steer = get_steering_vector("fact_recall", "comparison", layer)
    target_module = model.model.layers[layer]
    
    torch.manual_seed(42)
    v_rand = torch.randn_like(v_steer)
    v_rand = v_rand / torch.norm(v_rand) * torch.norm(v_steer)
    
    all_generated_texts = []
    
    print(f"\n--- OOD Robustness & Diversity Check (Fact->Comp, Layer 11, c=5.0) ---")
    for batch_idx, batch_prompts in enumerate(fact_batches):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        # Real Vector
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
        all_generated_texts.extend(outputs_real)
        buckets_real = [classify_fact_to_comparison(txt) for txt in outputs_real]
        b_rate_real = buckets_real.count('B') / len(buckets_real)
        
        # Random Vector
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
        buckets_rand = [classify_fact_to_comparison(txt) for txt in outputs_rand]
        b_rate_rand = buckets_rand.count('B') / len(buckets_rand)
        
        # McNemar
        n12 = n21 = 0
        for r_real, r_rand in zip(buckets_real, buckets_rand):
            if r_real == 'B' and r_rand != 'B': n12 += 1
            elif r_real != 'B' and r_rand == 'B': n21 += 1
        p_val = mcnemar_exact(n12, n21)
        
        print(f"Batch {batch_idx + 1} | Real B: {b_rate_real*100:.1f}% | Rand B: {b_rate_rand*100:.1f}% | p-val: {p_val:.4f}")
        
    print("\n--- Diversity Check ---")
    unique_strings = set([txt.strip() for txt in all_generated_texts])
    fraction_unique = len(unique_strings) / len(all_generated_texts)
    print(f"Total novel prompts: {len(all_generated_texts)}")
    print(f"Unique generated outputs: {len(unique_strings)}")
    print(f"Fraction unique: {fraction_unique*100:.1f}%")
    
    print("\n--- Sample Generations (first 15) ---")
    for i in range(15):
        print(f"Prompt: {all_prompts[i].split('Answer:')[0].strip()}")
        print(f"Output: '{all_generated_texts[i]}'\n")

if __name__ == "__main__":
    main()
