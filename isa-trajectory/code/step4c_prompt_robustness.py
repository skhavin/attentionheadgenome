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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    # Load original vector logic
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

    # Generate 4 disjoint batches of 20 NEW prompts
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
    for i in range(4):
        batch = [f"The capital of {c} is\nAnswer:" for c in countries[i*20:(i+1)*20]]
        fact_batches.append(batch)
        
    np.random.seed(42)
    arith_batches = []
    for i in range(4):
        batch = []
        for _ in range(20):
            x = np.random.randint(10, 99)
            y = np.random.randint(10, 99)
            batch.append(f"What is {x} + {y}?\nAnswer:")
        arith_batches.append(batch)

    # The top cells to replicate
    targets = [
        {"src": "arithmetic", "tgt": "sorting", "clf": classify_arithmetic_to_sorting, "layer": 27, "c": 1.5, "orig_real": 0.43, "batches": arith_batches},
        {"src": "fact_recall", "tgt": "comparison", "clf": classify_fact_to_comparison, "layer": 11, "c": 5.0, "orig_real": 0.83, "batches": fact_batches},
        {"src": "fact_recall", "tgt": "comparison", "clf": classify_fact_to_comparison, "layer": 25, "c": 3.0, "orig_real": 0.50, "batches": fact_batches},
        {"src": "fact_recall", "tgt": "comparison", "clf": classify_fact_to_comparison, "layer": 27, "c": 3.0, "orig_real": 0.73, "batches": fact_batches},
    ]

    for t in targets:
        src_cat, tgt_cat, classifier, layer, c = t["src"], t["tgt"], t["clf"], t["layer"], t["c"]
        print(f"\nPrompt Robustness for {src_cat}->{tgt_cat} Layer {layer} c={c} (Orig Real B: {t['orig_real']})")
        
        v_steer = get_steering_vector(src_cat, tgt_cat, layer)
        target_module = model.model.layers[layer]
        
        for batch_idx, batch_prompts in enumerate(t["batches"]):
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            
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
            buckets_real = [classifier(txt) for txt in outputs_real]
            b_rate = buckets_real.count('B') / len(buckets_real)
            print(f"  Batch {batch_idx + 1} (N=20 unseen prompts) | Hijack Rate: {b_rate:.2f}")

if __name__ == "__main__":
    main()
