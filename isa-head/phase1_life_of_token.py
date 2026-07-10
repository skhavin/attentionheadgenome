import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_first_token_id(tokenizer, text):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokens[0] if tokens else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", type=str, default="dataset_confirmation_20.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} on {DEVICE}")
    
    # 3050 constraints: bfloat16, device_map
    model_kwargs = {"device_map": DEVICE}
    if "gemma" in args.model_name.lower():
        try:
            model_kwargs["load_in_8bit"] = True
        except:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # For Phase 1, we use all 40 Discovery prompts
    print(f"Loaded {len(dataset)} Discovery prompts for Phase 1 (Life of a Token).")

    n_layers = model.config.num_hidden_layers
    unembed = model.lm_head.weight.detach().clone() # Shape: (Vocab, Hidden)
    final_norm = model.model.norm
    
    # Create shuffled targets for the control
    np.random.seed(42)
    shuffled_dataset = dataset.copy()
    np.random.shuffle(shuffled_dataset)
    
    results = []
    
    real_crossover_layers = []
    shuffled_crossover_layers = []

    for idx, item in enumerate(tqdm(dataset, desc="Phase 1: Life of a Token")):
        prompt = item["prompt"]
        target_str = item["target_full"] if item["task_type"] == "niah" else item["target"]
        target_id = get_first_token_id(tokenizer, target_str)
        
        # Control: shuffled target
        shuffled_target_str = shuffled_dataset[idx]["target_full"] if shuffled_dataset[idx]["task_type"] == "niah" else shuffled_dataset[idx]["target"]
        shuffled_target_id = get_first_token_id(tokenizer, shuffled_target_str)
        
        # If random shuffle gave same target, pick a default wrong one
        if shuffled_target_id == target_id:
            shuffled_target_id = get_first_token_id(tokenizer, " banana")
            
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Hook all layers to get r_l
        r_l_dict = {}
        def get_hook(l_idx):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                r_l_dict[l_idx] = hs[0, -1, :].detach().clone()
            return hook
            
        hooks = []
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(get_hook(i)))
            
        with torch.no_grad():
            _ = model(**tokens)
            
        for h in hooks:
            h.remove()
            
        # Metric: layer at which cosine(r_l, W_U[target]) reaches 90% of its final layer value
        target_vector = unembed[target_id].to(DEVICE).to(model.dtype)
        shuffled_vector = unembed[shuffled_target_id].to(DEVICE).to(model.dtype)
        
        r_L = final_norm(r_l_dict[n_layers - 1].unsqueeze(0))[0] # Final layer normalized
        final_target_cos = F.cosine_similarity(r_L.unsqueeze(0), target_vector.unsqueeze(0)).item()
        final_shuffled_cos = F.cosine_similarity(r_L.unsqueeze(0), shuffled_vector.unsqueeze(0)).item()
        
        real_crossover = n_layers - 1
        shuffled_crossover = n_layers - 1
        
        for l in range(n_layers):
            r_l = final_norm(r_l_dict[l].unsqueeze(0))[0]
            
            # Real Target
            cos_real = F.cosine_similarity(r_l.unsqueeze(0), target_vector.unsqueeze(0)).item()
            if cos_real >= 0.9 * final_target_cos and real_crossover == n_layers - 1:
                real_crossover = l
                
            # Shuffled Target (Control)
            # We check when it reaches 90% of the REAL target's final magnitude (as a threshold for "activation")
            cos_shuff = F.cosine_similarity(r_l.unsqueeze(0), shuffled_vector.unsqueeze(0)).item()
            if cos_shuff >= 0.9 * final_target_cos and shuffled_crossover == n_layers - 1:
                shuffled_crossover = l
                
        real_crossover_layers.append(real_crossover)
        shuffled_crossover_layers.append(shuffled_crossover)
        
        results.append({
            "prompt": prompt,
            "real_target": target_str,
            "shuffled_target": shuffled_target_str,
            "real_crossover_layer": real_crossover,
            "shuffled_crossover_layer": shuffled_crossover
        })
        
    print("\n--- Phase 1: Life of a Token Complete ---")
    print(f"Mean Real Crossover Layer: {np.mean(real_crossover_layers):.2f}")
    print(f"Mean Shuffled Crossover Layer (Control): {np.mean(shuffled_crossover_layers):.2f}")
    
    # Wilcoxon signed-rank test
    stat, p_val = stats.wilcoxon(real_crossover_layers, shuffled_crossover_layers, alternative='less')
    
    print("\n--- Pre-Registered Threshold Check ---")
    print(f"Wilcoxon p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("[RESULT] VALIDATED: The residual stream evolves towards the true target significantly earlier than a shuffled control.")
    else:
        print("[RESULT] FALSIFIED: The crossover effect is a general artifact of residual growth, not semantic targeting.")
        
    with open("phase1_results.json", "w", encoding="utf-8") as f:
        json.dump({"stats": {"mean_real": np.mean(real_crossover_layers), "mean_shuffled": np.mean(shuffled_crossover_layers), "p_val": p_val}, "raw": results}, f, indent=2)
        
if __name__ == "__main__":
    main()
