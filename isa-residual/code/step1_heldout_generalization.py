import json
import os
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_features(prompts, tokenizer):
    X = torch.zeros(len(prompts), 4)
    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]
        target_text = p["target"]
        
        tokens = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = tokens.input_ids.shape[1]
        target_len = len(target_text)
        num_digits = sum(c.isdigit() for c in prompt_text)
        density = num_digits / max(1, len(prompt_text))
        
        X[i, 0] = 1.0
        X[i, 1] = float(prompt_len)
        X[i, 2] = float(target_len)
        X[i, 3] = float(density)
        
    return X.to(DEVICE)

def extract_residuals(model, tokenizer, prompts, target_layer):
    residuals = []
    for item in prompts:
        tokens = tokenizer(item["prompt"], return_tensors="pt").to(DEVICE)
        cache = {}
        def hook_fn(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            if hidden.dim() == 3: cache["val"] = hidden[0, -1, :].detach().clone()
            else: cache["val"] = hidden[-1, :].detach().clone()
        handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**tokens)
        handle.remove()
        residuals.append(cache["val"].squeeze(0).float())
    return torch.stack(residuals)

def run_model_extraction(model_name, discovery, confirmation):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None, None, None, None

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    base_tasks = {"fact_recall", "niah", "copy", "pattern_induction", "counting", "comparison"}
    heldout_tasks = {"sorting", "arithmetic"}
    
    # Discovery (base 6)
    disc_filtered = [p for p in discovery if p["task_type"] in base_tasks]
    disc_res = extract_residuals(model, tokenizer, disc_filtered, target_layer)
    disc_X = extract_features(disc_filtered, tokenizer)
    
    # Normalize features based on discovery
    mean_X = disc_X[:, 1:].mean(dim=0)
    std_X = disc_X[:, 1:].std(dim=0) + 1e-8
    disc_X[:, 1:] = (disc_X[:, 1:] - mean_X) / std_X
    
    # Fit deconfounding on discovery
    W = torch.linalg.lstsq(disc_X, disc_res).solution
    disc_pure = disc_res - (disc_X @ W)
    
    # Confirmation (heldout 2)
    conf_filtered = [p for p in confirmation if p["task_type"] in heldout_tasks]
    conf_res = extract_residuals(model, tokenizer, conf_filtered, target_layer)
    conf_X = extract_features(conf_filtered, tokenizer)
    
    # Apply discovery normalization and deconfounding to confirmation
    conf_X[:, 1:] = (conf_X[:, 1:] - mean_X) / std_X
    conf_pure = conf_res - (conf_X @ W)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return disc_filtered, disc_pure.cpu(), conf_filtered, conf_pure.cpu()

def main():
    discovery = get_prompts("../../isa-head/dataset_discovery_224.json")
    confirmation = get_prompts("../../isa-head/dataset_confirmation_112.json")
    
    print("Running Step 1: Held-out operation generalization...")
    
    qwen_disc_p, qwen_disc_r, qwen_conf_p, qwen_conf_r = run_model_extraction("Qwen/Qwen2.5-1.5B", discovery, confirmation)
    llama_disc_p, llama_disc_r, llama_conf_p, llama_conf_r = run_model_extraction("unsloth/Llama-3.2-1B", discovery, confirmation)
    
    base_tasks = ["fact_recall", "niah", "copy", "pattern_induction", "counting", "comparison"]
    heldout_tasks = ["sorting", "arithmetic"]
    
    import torch.nn.functional as F
    
    def get_base_directions(prompts, residuals):
        dirs = {}
        for t in base_tasks:
            target_idx = [i for i, p in enumerate(prompts) if p["task_type"] == t]
            other_idx = [i for i, p in enumerate(prompts) if p["task_type"] != t]
            m_target = residuals[target_idx].mean(dim=0)
            m_other = residuals[other_idx].mean(dim=0)
            d = m_target - m_other
            dirs[t] = d / torch.norm(d)
        return dirs

    qwen_base_dirs = get_base_directions(qwen_disc_p, qwen_disc_r)
    llama_base_dirs = get_base_directions(llama_disc_p, llama_disc_r)
    
    results = {}
    
    for h_task in heldout_tasks:
        qwen_target_idx = [i for i, p in enumerate(qwen_conf_p) if p["task_type"] == h_task]
        llama_target_idx = [i for i, p in enumerate(llama_conf_p) if p["task_type"] == h_task]
        
        q_target_res = qwen_conf_r[qwen_target_idx]
        l_target_res = llama_conf_r[llama_target_idx]
        
        # We need an "other" to subtract to get the direction. We will use the discovery set mean as the general "other".
        q_other_mean = qwen_disc_r.mean(dim=0)
        l_other_mean = llama_disc_r.mean(dim=0)
        
        # Bootstrap
        n_boot = 10000
        corrs = []
        
        for b in range(n_boot):
            # Sample with replacement
            q_idx = np.random.choice(len(q_target_res), len(q_target_res), replace=True)
            l_idx = np.random.choice(len(l_target_res), len(l_target_res), replace=True)
            
            q_m = q_target_res[q_idx].mean(dim=0)
            l_m = l_target_res[l_idx].mean(dim=0)
            
            q_dir = q_m - q_other_mean
            q_dir = q_dir / torch.norm(q_dir)
            
            l_dir = l_m - l_other_mean
            l_dir = l_dir / torch.norm(l_dir)
            
            q_sims = [F.cosine_similarity(q_dir.unsqueeze(0), qwen_base_dirs[t].unsqueeze(0)).item() for t in base_tasks]
            l_sims = [F.cosine_similarity(l_dir.unsqueeze(0), llama_base_dirs[t].unsqueeze(0)).item() for t in base_tasks]
            
            corr, _ = stats.spearmanr(q_sims, l_sims)
            corrs.append(corr)
            
        corrs = np.array(corrs)
        corrs = corrs[~np.isnan(corrs)] # Remove NaNs in case of exact ties
        
        mean_corr = np.mean(corrs)
        ci_lower = np.percentile(corrs, 2.5)
        ci_upper = np.percentile(corrs, 97.5)
        
        results[h_task] = {
            "mean_spearman_rho": float(mean_corr),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "supports_generalization": float(ci_lower) > 0.0
        }
        
        print(f"\nHeld-out Task: {h_task}")
        print(f"Mean Correlation: {mean_corr:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Supports Generalization: {float(ci_lower) > 0.0}")

    os.makedirs("../outputs-isa-residual/step1", exist_ok=True)
    with open("../outputs-isa-residual/step1/step1_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
