import argparse
import json
import os
import sys
import torch
import scipy.stats as stats
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_first_token_id(tokenizer, text):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokens[0] if tokens else None

def cliffs_delta(x, y):
    """Calculates Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    
    gt = sum(1 for i in x for j in y if i > j)
    lt = sum(1 for i in x for j in y if i < j)
    
    return (gt - lt) / (n1 * n2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", type=str, default="dataset_confirmation_20.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    
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

    fact_prompts = [item for item in dataset if item["task_type"] == "fact_recall"]
    print(f"Loaded {len(fact_prompts)} Fact Recall prompts for MLP Confirmation.")

    # From Discovery Phase: Layer 24 is Boost-Correct, Layer 26 is Neutral
    target_layer_idx = 24
    control_layer_idx = 26
    
    target_drops = []
    control_drops = []

    for item in fact_prompts:
        prompt = item["prompt"]
        target_str = item["target"]
        target_id = get_first_token_id(tokenizer, target_str)
        
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # 1. Baseline Run
        with torch.no_grad():
            clean_outputs = model(**tokens)
            baseline_logit = clean_outputs.logits[0, -1, target_id].item()
            
        def get_ablation_hook():
            def hook(module, inp, out):
                return torch.zeros_like(out[0] if isinstance(out, tuple) else out)
            return hook
            
        # 2. Target Ablation (Boost-Correct MLP)
        layer_mlp = model.model.layers[target_layer_idx].mlp.down_proj
        target_hook = layer_mlp.register_forward_hook(get_ablation_hook())
        with torch.no_grad():
            ab_target_outputs = model(**tokens)
            ab_target_logit = ab_target_outputs.logits[0, -1, target_id].item()
        target_hook.remove()
        
        # 3. Control Ablation (Neutral MLP)
        control_mlp = model.model.layers[control_layer_idx].mlp.down_proj
        control_hook = control_mlp.register_forward_hook(get_ablation_hook())
        with torch.no_grad():
            ab_control_outputs = model(**tokens)
            ab_control_logit = ab_control_outputs.logits[0, -1, target_id].item()
        control_hook.remove()
        
        # Logit drop (positive means ablating it hurt the model)
        target_drop = baseline_logit - ab_target_logit
        control_drop = baseline_logit - ab_control_logit
        
        target_drops.append(target_drop)
        control_drops.append(control_drop)
        
    print(f"\n--- MLP Genome Confirmation Complete ({args.model_name}) ---")
    print(f"Target MLP (L24 - Boost-Correct) Mean Drop: {np.mean(target_drops):.4f}")
    print(f"Control MLP (L26 - Neutral) Mean Drop: {np.mean(control_drops):.4f}")
    
    # Statistical Tests
    if len(target_drops) > 0:
        # Wilcoxon signed-rank test
        try:
            stat, p_val = stats.wilcoxon(target_drops, control_drops, alternative='greater')
        except ValueError:
            # If differences are zero
            stat, p_val = 0, 1.0
            
        # Cliff's Delta
        c_delta = cliffs_delta(target_drops, control_drops)
        
        print("\n--- Pre-Registered Threshold Check ---")
        print(f"Cliff's Delta: {c_delta:.4f} (Threshold: > 0.33)")
        print(f"Wilcoxon p-value: {p_val:.4e} (Threshold: < 0.05)")
        
        if c_delta > 0.33 and p_val < 0.05:
            print("[RESULT] TAXONOMY VALIDATED! The Boost-Correct MLP is causally significant compared to the Neutral control.")
        else:
            print("[RESULT] TAXONOMY FALSIFIED. Fallback Logic Triggered: Label downgraded to 'Neutral'.")

if __name__ == "__main__":
    main()
