import argparse
import json
import os
import sys
import torch
import scipy.stats as stats
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--mode", type=str, choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} on {DEVICE} for Phase 2 ({args.mode})")
    
    model_kwargs = {"device_map": DEVICE}
    if "gemma" in args.model_name.lower():
        try:
            model_kwargs["load_in_8bit"] = True
        except:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_attentions=True, **model_kwargs)
    model.eval()

    dataset_file = "dataset_discovery_40.json" if args.mode == "discovery" else "dataset_confirmation_20.json"
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Use NIAH prompts to definitively prove long-range retrieval (rules out adjacent-token bias)
    prompts = [item for item in dataset if item["task_type"] == "niah"]
    print(f"Loaded {len(prompts)} NIAH prompts for Phase 2.")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    # We will track the metrics for all heads in the last 15% of layers
    target_layers = list(range(int(n_layers * 0.85), n_layers))
    
    head_scores = {l: {h: [] for h in range(n_heads)} for l in target_layers}
    
    all_target_attns = []
    all_uniform_attns = []
    all_distance_attns = []

    for item in tqdm(prompts, desc=f"Phase 2 ({args.mode})"):
        prompt = item["prompt"]
        password = item["password"] # e.g. "PW_657X"
        
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = tokens.input_ids[0].tolist()
        
        # Find the index of the password in the prompt
        # We can find the token sequence for the password
        pwd_tokens = tokenizer(password, add_special_tokens=False).input_ids
        
        target_idx = -1
        # Search for the sublist
        for i in range(len(input_ids) - len(pwd_tokens)):
            if input_ids[i:i+len(pwd_tokens)] == pwd_tokens:
                target_idx = i + len(pwd_tokens) - 1 # Use the last token of the password
                break
                
        if target_idx == -1:
            continue # Skip if we can't precisely locate it
            
        Q_idx = len(input_ids) - 1
        d = Q_idx - target_idx
        
        with torch.no_grad():
            outputs = model(**tokens)
            
        # outputs.attentions is a tuple of (batch, heads, seq, seq)
        for l in target_layers:
            attn = outputs.attentions[l][0] # (heads, seq, seq)
            
            for h in range(n_heads):
                # Attention from final query token to the target password token
                attn_target = attn[h, Q_idx, target_idx].item()
                
                # Uniform baseline
                attn_uniform = 1.0 / (Q_idx + 1)
                
                # Equal-distance baseline (Attention from Q-1 to T-1)
                if Q_idx - 1 >= 0 and target_idx - 1 >= 0:
                    attn_distance = attn[h, Q_idx - 1, target_idx - 1].item()
                else:
                    attn_distance = attn_uniform
                    
                if args.mode == "discovery":
                    # Score = how much it beats the positional bias and uniform bias
                    score = attn_target - max(attn_uniform, attn_distance)
                    head_scores[l][h].append(score)
                else:
                    # In confirmation, we just collect the raw values for the validated heads
                    # We will load the pre-registered heads below, but for now we'll collect everything
                    # and filter later.
                    pass
                    
        if args.mode == "confirmation":
            # For confirmation, we need to apply the pre-registered heads
            # Let's just store the arrays and we will test them
            all_target_attns.append(attn_target)
            all_uniform_attns.append(attn_uniform)
            all_distance_attns.append(attn_distance)
            
    if args.mode == "discovery":
        # Identify the top 5 Retrieval Heads
        flat_scores = []
        for l in target_layers:
            for h in range(n_heads):
                mean_score = np.mean(head_scores[l][h])
                flat_scores.append((mean_score, l, h))
                
        flat_scores.sort(reverse=True)
        top_heads = [{"layer": l, "head": h, "score": float(s)} for s, l, h in flat_scores[:5]]
        
        print("\n--- Phase 2 Discovery Complete ---")
        print("Top 5 Retrieval Heads (Q/K Circuit):")
        for th in top_heads:
            print(f"Layer {th['layer']}, Head {th['head']} (Score: {th['score']:.4f})")
            
        with open("phase2_retrieval_heads.json", "w") as f:
            json.dump(top_heads, f, indent=2)
            
    else:
        # CONFIRMATION MODE
        with open("phase2_retrieval_heads.json", "r") as f:
            registered_heads = json.load(f)
            
        print(f"\n--- Phase 2 Confirmation Complete ---")
        print(f"Validating {len(registered_heads)} pre-registered heads on N={len(prompts)} Confirmation prompts.")
        
        target_vals = []
        uniform_vals = []
        distance_vals = []
        
        for item in tqdm(prompts, desc="Confirming Heads"):
            prompt = item["prompt"]
            password = item["password"]
            tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = tokens.input_ids[0].tolist()
            pwd_tokens = tokenizer(password, add_special_tokens=False).input_ids
            
            target_idx = -1
            for i in range(len(input_ids) - len(pwd_tokens)):
                if input_ids[i:i+len(pwd_tokens)] == pwd_tokens:
                    target_idx = i + len(pwd_tokens) - 1
                    break
            if target_idx == -1: continue
            
            Q_idx = len(input_ids) - 1
            with torch.no_grad():
                outputs = model(**tokens)
                
            for rh in registered_heads:
                l, h = rh["layer"], rh["head"]
                attn = outputs.attentions[l][0]
                
                target_vals.append(attn[h, Q_idx, target_idx].item())
                uniform_vals.append(1.0 / (Q_idx + 1))
                if Q_idx - 1 >= 0 and target_idx - 1 >= 0:
                    distance_vals.append(attn[h, Q_idx - 1, target_idx - 1].item())
                else:
                    distance_vals.append(1.0 / (Q_idx + 1))
                    
        # Statistical Tests
        print(f"\nMean Target Attn: {np.mean(target_vals):.4f}")
        print(f"Mean Uniform Baseline: {np.mean(uniform_vals):.4f}")
        print(f"Mean Positional Baseline: {np.mean(distance_vals):.4f}")
        
        t_stat_uni, p_val_uni = stats.ttest_rel(target_vals, uniform_vals, alternative='greater')
        t_stat_dist, p_val_dist = stats.ttest_rel(target_vals, distance_vals, alternative='greater')
        
        print("\n--- Pre-Registered Threshold Check ---")
        print(f"Target vs Uniform p-value: {p_val_uni:.4e}")
        print(f"Target vs Positional p-value: {p_val_dist:.4e}")
        
        if p_val_uni < 0.05 and p_val_dist < 0.05:
            print("[RESULT] VALIDATED: The pre-registered Retrieval Heads demonstrably execute a semantic Q/K lookup, significantly beating both uniform and positional baselines on held-out data.")
        else:
            print("[RESULT] FALSIFIED: The heads fail to beat the controls on held-out data.")
            
        with open("phase2_results.json", "w") as f:
            json.dump({
                "mean_target": float(np.mean(target_vals)),
                "mean_uniform": float(np.mean(uniform_vals)),
                "mean_positional": float(np.mean(distance_vals)),
                "p_uni": p_val_uni,
                "p_dist": p_val_dist
            }, f, indent=2)

if __name__ == "__main__":
    main()
