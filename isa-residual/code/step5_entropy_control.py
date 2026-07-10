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

def extract_features_and_residuals(model, tokenizer, prompts, target_layer):
    X = torch.zeros(len(prompts), 6)
    residuals = []
    
    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]
        target_text = p["target"]
        
        tokens = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        prompt_len = tokens.input_ids.shape[1]
        target_len = len(target_text)
        num_digits = sum(c.isdigit() for c in prompt_text)
        density = num_digits / max(1, len(prompt_text))
        
        cache = {}
        def hook_fn(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            if hidden.dim() == 3: cache["val"] = hidden[0, -1, :].detach().clone()
            else: cache["val"] = hidden[-1, :].detach().clone()
            
        handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            outputs = model(**tokens)
            
        handle.remove()
        residuals.append(cache["val"].squeeze(0).float())
        
        # Calculate Entropy and Top-1 Prob for the last token position
        logits = outputs.logits[0, -1, :].float()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        top1_prob = probs.max().item()
        
        X[i, 0] = 1.0 # Bias
        X[i, 1] = float(prompt_len)
        X[i, 2] = float(target_len)
        X[i, 3] = float(density)
        X[i, 4] = float(entropy)
        X[i, 5] = float(top1_prob)
        
    for j in range(1, 6):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-8)
        
    return torch.stack(residuals).cpu(), X.cpu()

def run_model_analysis(model_name, discovery):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    Y, X = extract_features_and_residuals(model, tokenizer, discovery, target_layer)
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return Y, X

def compute_similarity_matrix(Y, discovery, tasks_to_use):
    directions = {}
    for t in tasks_to_use:
        target_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        other_idx = [i for i, p in enumerate(discovery) if p["task_type"] != t]
        mean_target = Y[target_idx].mean(dim=0)
        mean_other = Y[other_idx].mean(dim=0)
        direction = mean_target - mean_other
        directions[t] = direction / torch.norm(direction)
        
    sim_matrix = np.zeros((len(tasks_to_use), len(tasks_to_use)))
    import torch.nn.functional as F
    for i in range(len(tasks_to_use)):
        for j in range(len(tasks_to_use)):
            if i != j:
                sim = F.cosine_similarity(directions[tasks_to_use[i]].unsqueeze(0), directions[tasks_to_use[j]].unsqueeze(0)).item()
                sim_matrix[i, j] = sim
                
    return sim_matrix

def compute_overall_rho(q_mat, l_mat):
    n = q_mat.shape[0]
    idx = np.triu_indices(n, k=1)
    q_vec = q_mat[idx]
    l_vec = l_mat[idx]
    corr, _ = stats.spearmanr(q_vec, l_vec)
    return corr

def main():
    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    print("Running Final 12-Category Scale-up RSA with Entropy/Difficulty Control...")
    
    all_tasks = [
        "comparison", "sorting", "arithmetic", "counting",
        "fact_recall", "niah", "copy", "pattern_induction", "negation", "set_membership", "entailment", "concatenation"
    ]
    
    Y_qwen, X_qwen = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    Y_llama, X_llama = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    
    # 1. Base Confound Regression (3 Features: Length, Target Length, Density)
    X_qwen_base = X_qwen[:, :4]
    X_llama_base = X_llama[:, :4]
    
    W_qwen_base = torch.linalg.lstsq(X_qwen_base, Y_qwen).solution
    Y_qwen_base = Y_qwen - (X_qwen_base @ W_qwen_base)
    
    W_llama_base = torch.linalg.lstsq(X_llama_base, Y_llama).solution
    Y_llama_base = Y_llama - (X_llama_base @ W_llama_base)
    
    q_mat_base = compute_similarity_matrix(Y_qwen_base, discovery, all_tasks)
    l_mat_base = compute_similarity_matrix(Y_llama_base, discovery, all_tasks)
    rho_base = compute_overall_rho(q_mat_base, l_mat_base)
    
    # 2. Full Entropy Control Regression (5 Features: + Entropy, + Top1 Prob)
    W_qwen_full = torch.linalg.lstsq(X_qwen, Y_qwen).solution
    Y_qwen_full = Y_qwen - (X_qwen @ W_qwen_full)
    
    W_llama_full = torch.linalg.lstsq(X_llama, Y_llama).solution
    Y_llama_full = Y_llama - (X_llama @ W_llama_full)
    
    q_mat_full = compute_similarity_matrix(Y_qwen_full, discovery, all_tasks)
    l_mat_full = compute_similarity_matrix(Y_llama_full, discovery, all_tasks)
    rho_full = compute_overall_rho(q_mat_full, l_mat_full)
    
    # Calculate Mantel p-value for the fully controlled matrix
    n = len(all_tasks)
    idx = np.triu_indices(n, k=1)
    q_vec_full = q_mat_full[idx]
    
    greater_count = 0
    n_perm = 10000
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        l_mat_perm = l_mat_full[perm_idx, :][:, perm_idx]
        l_vec_perm = l_mat_perm[idx]
        corr, _ = stats.spearmanr(q_vec_full, l_vec_perm)
        if corr >= rho_full:
            greater_count += 1
            
    p_value = (greater_count + 1) / (n_perm + 1)
    
    print(f"\n--- Entropy & Difficulty Control Results ---")
    print(f"Base Rho (Length/Density): {rho_base:.4f}")
    print(f"Entropy-Controlled Rho (Full 5 Features): {rho_full:.4f}")
    print(f"Mantel P-Value (Entropy-Controlled): {p_value:.5f}")
    
    survival_ratio = rho_full / rho_base if rho_base > 0 else 0
    print(f"Structural Survival Ratio: {survival_ratio*100:.2f}%")

if __name__ == "__main__":
    main()
