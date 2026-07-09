import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    with open("dataset_confirmation_20.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # ONLY FACT RECALL (N=7)
    fact_prompts = [item for item in dataset if item["task_type"] == "fact_recall"]
    
    pairs = []
    for i in range(len(fact_prompts)):
        clean = fact_prompts[i]
        corrupted = fact_prompts[(i + 1) % len(fact_prompts)]
        t_clean = tokenizer(clean["target"], add_special_tokens=False).input_ids[0]
        t_corr = tokenizer(corrupted["target"], add_special_tokens=False).input_ids[0]
        pairs.append((clean["prompt"], corrupted["prompt"], t_clean, t_corr))

    test_layers = [22, 25, 27]
    layer_restorations = {l: [] for l in test_layers}
    layer_placebo_restorations = {l: [] for l in test_layers}

    def get_logit_diff(logits, target_id, corrupted_id):
        return (logits[0, -1, target_id] - logits[0, -1, corrupted_id]).item()

    for clean_prompt, corrupted_prompt, t_clean, t_corr in pairs:
        tok_clean = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
        tok_corr = tokenizer(corrupted_prompt, return_tensors="pt").to(DEVICE)
        
        clean_mlp_cache = {}
        def clean_cache_hook(module, args, output, layer_idx):
            clean_mlp_cache[layer_idx] = output[0, -1, :].detach().clone()
            return output
            
        handles = []
        for l in range(model.config.num_hidden_layers):
            h = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o, l_idx=l: clean_cache_hook(m, a, o, l_idx)
            )
            handles.append(h)
            
        with torch.no_grad():
            clean_logits = model(**tok_clean).logits
        for h in handles: h.remove()
        
        clean_diff = get_logit_diff(clean_logits, t_clean, t_corr)
        with torch.no_grad():
            corr_logits = model(**tok_corr).logits
        corr_diff = get_logit_diff(corr_logits, t_clean, t_corr)
        
        diff_denom = clean_diff - corr_diff
        if diff_denom < 1e-4: continue
            
        for l in test_layers:
            def patch_hook(module, args, output, patch_tensor):
                output[0, -1, :] = patch_tensor
                return output
                
            h_true = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o: patch_hook(m, a, o, clean_mlp_cache[l])
            )
            with torch.no_grad(): patched_logits = model(**tok_corr).logits
            h_true.remove()
            
            patched_diff = get_logit_diff(patched_logits, t_clean, t_corr)
            layer_restorations[l].append((patched_diff - corr_diff) / diff_denom)
            
            placebo_l = l - 5 if l >= 5 else l + 5
            h_placebo = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o: patch_hook(m, a, o, clean_mlp_cache[placebo_l])
            )
            with torch.no_grad(): placebo_logits = model(**tok_corr).logits
            h_placebo.remove()
            
            placebo_diff = get_logit_diff(placebo_logits, t_clean, t_corr)
            layer_placebo_restorations[l].append((placebo_diff - corr_diff) / diff_denom)
            
    print("\n--- Fact Recall Only Check (N=7) ---")
    for l in test_layers:
        trues = layer_restorations[l]
        placebos = layer_placebo_restorations[l]
        mean_true = np.mean(trues)
        mean_placebo = np.mean(placebos)
        try:
            w_stat, p_val = stats.wilcoxon(trues, placebos, alternative='greater')
        except:
            p_val = 1.0
        print(f"Layer {l}: True {mean_true*100:.1f}% | Placebo {mean_placebo*100:.1f}% | p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()
