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
        
        X[i, 0] = 1.0 # Bias
        X[i, 1] = float(prompt_len)
        X[i, 2] = float(target_len)
        X[i, 3] = float(density)
        
    for j in range(1, 4):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-8)
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

def run_model_analysis(model_name, discovery):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    Y = extract_residuals(model, tokenizer, discovery, target_layer).cpu()
    X = extract_features(discovery, tokenizer).cpu()
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return Y, X

def compute_similarity_matrix(Y, discovery, tasks_to_use):
    directions = {}
    for t in tasks_to_use:
        target_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        other_idx = [i for i, p in enumerate(discovery) if p["task_type"] != t] # other vs ALL 12, standard RSA
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

def get_cross_block_rho(q_mat_full, l_mat_full, numeric_idx, symbolic_idx):
    # Extract only the pairwise similarities BETWEEN numeric and symbolic categories
    q_cross = []
    l_cross = []
    for i in numeric_idx:
        for j in symbolic_idx:
            q_cross.append(q_mat_full[i, j])
            l_cross.append(l_mat_full[i, j])
    corr, _ = stats.spearmanr(q_cross, l_cross)
    return corr

def main():
    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    print("Running Final 12-Category Scale-up RSA...")
    
    Y_qwen, X_qwen = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    Y_llama, X_llama = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    
    # Global Deconfounding
    W_qwen = torch.linalg.lstsq(X_qwen, Y_qwen).solution
    Y_qwen_pure = Y_qwen - (X_qwen @ W_qwen)
    
    W_llama = torch.linalg.lstsq(X_llama, Y_llama).solution
    Y_llama_pure = Y_llama - (X_llama @ W_llama)
    
    all_tasks = [
        "comparison", "sorting", "arithmetic", "counting", # Numeric (4)
        "fact_recall", "niah", "copy", "pattern_induction", "negation", "set_membership", "entailment", "concatenation" # Symbolic (8)
    ]
    numeric_tasks = all_tasks[:4]
    symbolic_tasks = all_tasks[4:]
    
    q_mat_full = compute_similarity_matrix(Y_qwen_pure, discovery, all_tasks)
    l_mat_full = compute_similarity_matrix(Y_llama_pure, discovery, all_tasks)
    
    q_mat_num = compute_similarity_matrix(Y_qwen_pure, discovery, numeric_tasks)
    l_mat_num = compute_similarity_matrix(Y_llama_pure, discovery, numeric_tasks)
    
    q_mat_sym = compute_similarity_matrix(Y_qwen_pure, discovery, symbolic_tasks)
    l_mat_sym = compute_similarity_matrix(Y_llama_pure, discovery, symbolic_tasks)
    
    # 1. Block Correlations
    rho_num = compute_overall_rho(q_mat_num, l_mat_num)
    rho_sym = compute_overall_rho(q_mat_sym, l_mat_sym)
    
    numeric_idx = list(range(4))
    symbolic_idx = list(range(4, 12))
    rho_cross = get_cross_block_rho(q_mat_full, l_mat_full, numeric_idx, symbolic_idx)
    
    print(f"\n--- Subgroup Block Analysis ---")
    print(f"Numeric Block Internal Rho (N=4): {rho_num:.4f}")
    print(f"Symbolic Block Internal Rho (N=8): {rho_sym:.4f}")
    print(f"Numeric-vs-Symbolic Cross-Block Rho: {rho_cross:.4f}")
    
    # 2. Aggregate Check
    rho_global = compute_overall_rho(q_mat_full, l_mat_full)
    print(f"\n--- Aggregate Global RSA (N=12) ---")
    print(f"Global Rho: {rho_global:.4f}")
    
    # Mantel Test (10,000 permutations)
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
        if corr >= rho_global:
            greater_count += 1
            
    p_value = (greater_count + 1) / (n_perm + 1)
    print(f"Mantel P-Value: {p_value:.5f}")
    
    os.makedirs("../outputs-isa-residual/step4_scaleup", exist_ok=True)
    with open("../outputs-isa-residual/step4_scaleup/rsa_results.json", "w") as f:
        json.dump({
            "rho_numeric_internal": float(rho_num),
            "rho_symbolic_internal": float(rho_sym),
            "rho_cross_block": float(rho_cross),
            "rho_global": float(rho_global),
            "mantel_p_value": float(p_value)
        }, f, indent=2)

if __name__ == "__main__":
    main()
