import argparse
import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--mode", type=str, choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} on {DEVICE} for Phase 3 ({args.mode})")
    
    model_kwargs = {"device_map": DEVICE, "torch_dtype": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    dataset_file = "dataset_discovery_40.json" if args.mode == "discovery" else "dataset_confirmation_20.json"
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Group by task_type to create valid corrupted pairs within the same task
    task_groups = {}
    for item in dataset:
        tt = item["task_type"]
        if tt not in task_groups: task_groups[tt] = []
        task_groups[tt].append(item)
        
    pairs = []
    for tt, items in task_groups.items():
        if len(items) < 2: continue
        for i in range(len(items)):
            clean = items[i]
            corrupted = items[(i + 1) % len(items)]
            
            # For NIAH, target is 'target_full'. For others, it's 'target'.
            tgt_clean = clean.get("target_full", clean.get("target"))
            tgt_corr = corrupted.get("target_full", corrupted.get("target"))
            
            t_clean = tokenizer(tgt_clean, add_special_tokens=False).input_ids[0]
            t_corr = tokenizer(tgt_corr, add_special_tokens=False).input_ids[0]
            
            pairs.append((clean["prompt"], corrupted["prompt"], t_clean, t_corr))
            
    print(f"Loaded {len(pairs)} pairs for Phase 3.")

    for clean_prompt, corrupted_prompt, t_clean, t_corr in tqdm(pairs, desc=f"Phase 3 ({args.mode})"):
        
        tok_clean = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
        tok_corr = tokenizer(corrupted_prompt, return_tensors="pt").to(DEVICE)
        
        # Ensure lengths match at the end (we only patch the last token)
        # For Fact Recall, they are usually identical length or we just patch index -1.
        
        # 1. Run Clean and Cache ALL MLP outputs at the last token
        clean_mlp_cache = {}
        def clean_cache_hook(module, args, output, layer_idx):
            # output is (batch, seq, hidden)
            clean_mlp_cache[layer_idx] = output[0, -1, :].detach().clone()
            return output
            
        handles = []
        for l in range(n_layers):
            h = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o, l_idx=l: clean_cache_hook(m, a, o, l_idx)
            )
            handles.append(h)
            
        with torch.no_grad():
            clean_logits = model(**tok_clean).logits
            
        for h in handles: h.remove()
        
        clean_diff = get_logit_diff(clean_logits, t_clean, t_corr)
        
        # 2. Run Corrupted Baseline
        with torch.no_grad():
            corr_logits = model(**tok_corr).logits
        corr_diff = get_logit_diff(corr_logits, t_clean, t_corr)
        
        diff_denom = clean_diff - corr_diff
        if diff_denom < 1e-4: # Skip if the prompt pair doesn't work well
            continue
            
        # 3. Patching Loop
        for l in test_layers:
            # TRUE PATCH: Inject clean_mlp_cache[l] into layer l of corrupted run
            def patch_hook(module, args, output, patch_tensor):
                output[0, -1, :] = patch_tensor
                return output
                
            h_true = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o: patch_hook(m, a, o, clean_mlp_cache[l])
            )
            
            with torch.no_grad():
                patched_logits = model(**tok_corr).logits
            h_true.remove()
            
            patched_diff = get_logit_diff(patched_logits, t_clean, t_corr)
            restoration = (patched_diff - corr_diff) / diff_denom
            layer_restorations[l].append(restoration)
            
            # PLACEBO PATCH: Inject clean_mlp_cache[l-5] (or another random non-adjacent layer)
            placebo_l = l - 5 if l >= 5 else l + 5
            h_placebo = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o: patch_hook(m, a, o, clean_mlp_cache[placebo_l])
            )
            
            with torch.no_grad():
                placebo_logits = model(**tok_corr).logits
            h_placebo.remove()
            
            placebo_diff = get_logit_diff(placebo_logits, t_clean, t_corr)
            placebo_rest = (placebo_diff - corr_diff) / diff_denom
            layer_placebo_restorations[l].append(placebo_rest)
            
    # Compute Statistics
    if args.mode == "discovery":
        print("\n--- Phase 3 Discovery Complete ---")
        results = []
        for l in test_layers:
            mean_rest = np.mean(layer_restorations[l])
            results.append((mean_rest, l))
            
        results.sort(reverse=True)
        top_mlps = [l for _, l in results[:3]]
        
        print("Top 3 Query-Generating MLPs identified:")
        for r, l in results[:3]:
            print(f"Layer {l}: {r*100:.1f}% logit-diff restoration")
            
        with open("phase3_query_mlps.json", "w") as f:
            json.dump(top_mlps, f, indent=2)
            
    else:
        with open("phase3_query_mlps.json", "r") as f:
            top_mlps = json.load(f)
            
        print("\n--- Phase 3 Confirmation Complete ---")
        p_values = {}
        for l in top_mlps:
            trues = layer_restorations[l]
            placebos = layer_placebo_restorations[l]
            
            mean_true = np.mean(trues)
            mean_placebo = np.mean(placebos)
            
            # Wilcoxon signed-rank
            try:
                w_stat, p_val = stats.wilcoxon(trues, placebos, alternative='greater')
            except ValueError:
                p_val = 1.0 # If all differences are zero
                
            print(f"\nMLP Layer {l}:")
            print(f"  True Restoration: {mean_true*100:.1f}%")
            print(f"  Placebo Restoration: {mean_placebo*100:.1f}%")
            print(f"  p-value: {p_val:.4e}")
            
            p_values[l] = {
                "mean_true": float(mean_true),
                "mean_placebo": float(mean_placebo),
                "p_val": p_val
            }
            
        with open("phase3_results.json", "w") as f:
            json.dump(p_values, f, indent=2)

if __name__ == "__main__":
    main()
