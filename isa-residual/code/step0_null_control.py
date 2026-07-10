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

def compute_similarity_matrix(Y, discovery):
    task_types = list(set([p["task_type"] for p in discovery]))
    task_types.sort()
    directions = {}
    for t in task_types:
        target_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        other_idx = [i for i, p in enumerate(discovery) if p["task_type"] != t]
        mean_target = Y[target_idx].mean(dim=0)
        mean_other = Y[other_idx].mean(dim=0)
        direction = mean_target - mean_other
        directions[t] = direction / torch.norm(direction)
        
    sim_matrix = np.zeros((len(task_types), len(task_types)))
    import torch.nn.functional as F
    for i in range(len(task_types)):
        for j in range(len(task_types)):
            if i != j:
                sim = F.cosine_similarity(directions[task_types[i]].unsqueeze(0), directions[task_types[j]].unsqueeze(0)).item()
                sim_matrix[i, j] = sim
                
    return sim_matrix, task_types

def compute_overall_rho(q_mat, l_mat):
    n = q_mat.shape[0]
    idx = np.triu_indices(n, k=1)
    q_vec = q_mat[idx]
    l_vec = l_mat[idx]
    corr, _ = stats.spearmanr(q_vec, l_vec)
    return corr

def main():
    discovery = get_prompts("../../isa-head/dataset_discovery_224.json")
    print("Running Null Regression Control...")
    
    Y_qwen, X_qwen = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    Y_llama, X_llama = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    
    # 1. Real regression (Sanity Check)
    W_qwen = torch.linalg.lstsq(X_qwen, Y_qwen).solution
    Y_qwen_pure = Y_qwen - (X_qwen @ W_qwen)
    
    W_llama = torch.linalg.lstsq(X_llama, Y_llama).solution
    Y_llama_pure = Y_llama - (X_llama @ W_llama)
    
    q_mat_pure, _ = compute_similarity_matrix(Y_qwen_pure, discovery)
    l_mat_pure, _ = compute_similarity_matrix(Y_llama_pure, discovery)
    
    true_rho = compute_overall_rho(q_mat_pure, l_mat_pure)
    
    # Analyze digit-density weights (column 3 in X)
    # The weights for digit density across the 1536 dimensions
    w_digit_qwen = W_qwen[3, :]
    print(f"True Deconfounded Rho: {true_rho:.4f}")
    print(f"Digit-Density Weights Magnitude (Qwen): {torch.norm(w_digit_qwen).item():.4f}")
    
    # 2. 50-Shuffle Control
    n_shuffles = 50
    null_rhos = []
    
    for s in range(n_shuffles):
        # Shuffle the rows of X together to preserve joint covariate structure but break residual link
        perm_idx = torch.randperm(X_qwen.shape[0])
        X_qwen_shuffled = X_qwen[perm_idx, :]
        X_llama_shuffled = X_llama[perm_idx, :] # keep the same permutation for both so they are parallel
        
        W_qwen_shuf = torch.linalg.lstsq(X_qwen_shuffled, Y_qwen).solution
        Y_qwen_null = Y_qwen - (X_qwen_shuffled @ W_qwen_shuf)
        
        W_llama_shuf = torch.linalg.lstsq(X_llama_shuffled, Y_llama).solution
        Y_llama_null = Y_llama - (X_llama_shuffled @ W_llama_shuf)
        
        q_mat_null, _ = compute_similarity_matrix(Y_qwen_null, discovery)
        l_mat_null, _ = compute_similarity_matrix(Y_llama_null, discovery)
        
        null_rho = compute_overall_rho(q_mat_null, l_mat_null)
        null_rhos.append(null_rho)
        
    null_rhos = np.array(null_rhos)
    print(f"\nNull Distribution (N=50 shuffles):")
    print(f"Mean Rho: {np.mean(null_rhos):.4f}")
    print(f"Max Rho:  {np.max(null_rhos):.4f}")
    print(f"Min Rho:  {np.min(null_rhos):.4f}")
    
    os.makedirs("../outputs-isa-residual/step0_null", exist_ok=True)
    with open("../outputs-isa-residual/step0_null/null_control_results.json", "w") as f:
        json.dump({
            "true_rho": float(true_rho),
            "digit_density_weight_norm": float(torch.norm(w_digit_qwen).item()),
            "null_rhos_distribution": null_rhos.tolist(),
            "null_mean": float(np.mean(null_rhos)),
            "null_max": float(np.max(null_rhos))
        }, f, indent=2)

if __name__ == "__main__":
    main()
