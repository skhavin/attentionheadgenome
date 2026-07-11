import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import scipy.stats as stats

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
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    print(f"Extracting representations for {model_name} at layer {target_layer}...")
    print("Extracting 5 Covariates: Prompt Length, Target Length, Numeric Density, Entropy, Top-1 Confidence")
    Y, X = extract_features_and_residuals(model, tokenizer, discovery, target_layer)
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return Y, X

def deconfound(Y, X):
    print("Executing 5-Covariate Full Deconfounding (Length, Target, Density, Entropy, Confidence)...")
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    W = torch.linalg.lstsq(X, Y).solution
    Y_pure = Y - (X @ W)
    return Y_pure.cpu()

def compute_centroids(Y, discovery, all_tasks):
    centroids = []
    for t in all_tasks:
        target_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        mean_target = Y[target_idx].mean(dim=0)
        centroids.append(mean_target)
    return torch.stack(centroids)

def linear_cka(X, Y):
    # X: (N, D1), Y: (N, D2)
    # Center columns
    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)
    
    # Linear CKA: ||X_c X_c^T * Y_c Y_c^T||_F / (||X_c X_c^T||_F * ||Y_c Y_c^T||_F)
    # For small N (like 12), computing the NxN Gram matrix is faster and numerically stable
    gram_x = X_c @ X_c.T
    gram_y = Y_c @ Y_c.T
    
    dot_prod = torch.sum(gram_x * gram_y)
    norm_x = torch.sqrt(torch.sum(gram_x * gram_x))
    norm_y = torch.sqrt(torch.sum(gram_y * gram_y))
    
    return (dot_prod / (norm_x * norm_y)).item()

def cka_permutation_test(X_centroids, Y_centroids, true_cka, n_permutations=10000):
    N = X_centroids.shape[0]
    count = 0
    Y_c = Y_centroids - Y_centroids.mean(dim=0)
    gram_y = Y_c @ Y_c.T
    
    X_c = X_centroids - X_centroids.mean(dim=0)
    gram_x = X_c @ X_c.T
    norm_x = torch.sqrt(torch.sum(gram_x * gram_x))
    
    for _ in range(n_permutations):
        idx = torch.randperm(N)
        shuffled_gram_y = gram_y[idx][:, idx]
        
        dot_prod = torch.sum(gram_x * shuffled_gram_y)
        norm_y = torch.sqrt(torch.sum(shuffled_gram_y * shuffled_gram_y))
        
        shuffled_cka = (dot_prod / (norm_x * norm_y)).item()
        if shuffled_cka >= true_cka:
            count += 1
            
    p_value = count / n_permutations
    if p_value == 0:
        p_value = 1.0 / n_permutations
    return p_value

def main():
    print("Running Linear CKA Architectural Universality Test (Strict Centroid Parity)...")
    
    # Threshold for success
    print("Pre-registered Threshold for Success: CKA score p-value < 0.05")
    
    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    
    all_tasks = [
        "comparison", "sorting", "arithmetic", "counting",
        "fact_recall", "niah", "copy", "pattern_induction", "negation", "set_membership", "entailment", "concatenation"
    ]
    
    # 1. Qwen
    Y_qwen, X_qwen = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    Y_qwen_pure = deconfound(Y_qwen, X_qwen)
    qwen_centroids = compute_centroids(Y_qwen_pure, discovery, all_tasks)
    
    # 2. Llama
    Y_llama, X_llama = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    Y_llama_pure = deconfound(Y_llama, X_llama)
    llama_centroids = compute_centroids(Y_llama_pure, discovery, all_tasks)
    
    # 3. Phi-1.5
    Y_phi, X_phi = run_model_analysis("microsoft/phi-1_5", discovery)
    Y_phi_pure = deconfound(Y_phi, X_phi)
    phi_centroids = compute_centroids(Y_phi_pure, discovery, all_tasks)
    
    print("\n--- CKA on Deconfounded 12-Category Centroids (Pure Computation) ---")
    
    # Qwen <-> Llama
    cka_q_l = linear_cka(qwen_centroids, llama_centroids)
    p_q_l = cka_permutation_test(qwen_centroids, llama_centroids, cka_q_l)
    print(f"Qwen <-> Llama: CKA = {cka_q_l:.4f} (p = {p_q_l:.5f})")
    
    # Qwen <-> Phi-1.5
    cka_q_p = linear_cka(qwen_centroids, phi_centroids)
    p_q_p = cka_permutation_test(qwen_centroids, phi_centroids, cka_q_p)
    print(f"Qwen <-> Phi-1.5: CKA = {cka_q_p:.4f} (p = {p_q_p:.5f})")
    
    # Llama <-> Phi-1.5
    cka_l_p = linear_cka(llama_centroids, phi_centroids)
    p_l_p = cka_permutation_test(llama_centroids, phi_centroids, cka_l_p)
    print(f"Llama <-> Phi-1.5: CKA = {cka_l_p:.4f} (p = {p_l_p:.5f})")
    
    os.makedirs("../outputs-isa-residual/step7b_cka", exist_ok=True)
    with open("../outputs-isa-residual/step7b_cka/cka_results.json", "w") as f:
        json.dump({
            "deconfounded_centroids": {
                "qwen_llama": {"cka": float(cka_q_l), "p_value": float(p_q_l)},
                "qwen_phi": {"cka": float(cka_q_p), "p_value": float(p_q_p)},
                "llama_phi": {"cka": float(cka_l_p), "p_value": float(p_l_p)}
            }
        }, f, indent=2)

if __name__ == "__main__":
    main()
