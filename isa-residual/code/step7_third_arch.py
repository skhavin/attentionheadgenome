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
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    print(f"Extracting representations for {model_name} at layer {target_layer}...")
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

def compute_overall_rho(mat1, mat2):
    n = mat1.shape[0]
    idx = np.triu_indices(n, k=1)
    vec1 = mat1[idx]
    vec2 = mat2[idx]
    corr, _ = stats.spearmanr(vec1, vec2)
    return corr

def run_mantel_test(mat1, mat2, rho_obs):
    n = mat1.shape[0]
    idx = np.triu_indices(n, k=1)
    vec1 = mat1[idx]
    
    greater_count = 0
    n_perm = 10000
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        mat2_perm = mat2[perm_idx, :][:, perm_idx]
        vec2_perm = mat2_perm[idx]
        corr, _ = stats.spearmanr(vec1, vec2_perm)
        if corr >= rho_obs:
            greater_count += 1
            
    p_value = (greater_count + 1) / (n_perm + 1)
    return p_value

def deconfound(Y, X):
    W = torch.linalg.lstsq(X, Y).solution
    return Y - (X @ W)

def main():
    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    print("Running 3-way Architectural Universality Test (Step 7)...")
    
    all_tasks = [
        "comparison", "sorting", "arithmetic", "counting",
        "fact_recall", "niah", "copy", "pattern_induction", "negation", "set_membership", "entailment", "concatenation"
    ]
    
    # 1. Qwen
    Y_qwen, X_qwen = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    Y_qwen_pure = deconfound(Y_qwen, X_qwen)
    mat_qwen = compute_similarity_matrix(Y_qwen_pure, discovery, all_tasks)
    
    # 2. Llama
    Y_llama, X_llama = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    Y_llama_pure = deconfound(Y_llama, X_llama)
    mat_llama = compute_similarity_matrix(Y_llama_pure, discovery, all_tasks)
    
    # 3. Phi-1.5 (Ungated, highly distinct architecture - Microsoft Phi)
    Y_phi, X_phi = run_model_analysis("microsoft/phi-1_5", discovery)
    Y_phi_pure = deconfound(Y_phi, X_phi)
    mat_phi = compute_similarity_matrix(Y_phi_pure, discovery, all_tasks)
    
    # Pairwise calculations
    rho_q_l = compute_overall_rho(mat_qwen, mat_llama)
    p_q_l = run_mantel_test(mat_qwen, mat_llama, rho_q_l)
    
    rho_q_p = compute_overall_rho(mat_qwen, mat_phi)
    p_q_p = run_mantel_test(mat_qwen, mat_phi, rho_q_p)
    
    rho_l_p = compute_overall_rho(mat_llama, mat_phi)
    p_l_p = run_mantel_test(mat_llama, mat_phi, rho_l_p)
    
    print("\n--- Step 7: Tripartite Structural Universality Results ---")
    print(f"Qwen <-> Llama: Rho = {rho_q_l:.4f} (p = {p_q_l:.5f})")
    print(f"Qwen <-> Phi-1.5: Rho = {rho_q_p:.4f} (p = {p_q_p:.5f})")
    print(f"Llama <-> Phi-1.5: Rho = {rho_l_p:.4f} (p = {p_l_p:.5f})")
    
    os.makedirs("../outputs-isa-residual/step7_third_arch", exist_ok=True)
    with open("../outputs-isa-residual/step7_third_arch/tripartite_results.json", "w") as f:
        json.dump({
            "qwen_llama": {"rho": float(rho_q_l), "p_value": float(p_q_l)},
            "qwen_phi": {"rho": float(rho_q_p), "p_value": float(p_q_p)},
            "llama_phi": {"rho": float(rho_l_p), "p_value": float(p_l_p)}
        }, f, indent=2)

if __name__ == "__main__":
    main()
