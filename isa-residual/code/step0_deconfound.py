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

def deconfound_residuals(Y, X):
    W = torch.linalg.lstsq(X, Y).solution
    Y_pred = X @ W
    return Y - Y_pred

def run_model_analysis(model_name, discovery):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    residuals = []
    for item in discovery:
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
        
    Y = torch.stack(residuals).to(DEVICE)
    X = extract_features(discovery, tokenizer)
    Y_pure = deconfound_residuals(Y, X)
    
    res_dict = {p["id"]: Y_pure[i].cpu() for i, p in enumerate(discovery)}

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    task_types = list(set([p["task_type"] for p in discovery]))
    task_types.sort()
    directions = {}
    for t in task_types:
        target_vecs = [res_dict[p["id"]] for p in discovery if p["task_type"] == t]
        other_vecs = [res_dict[p["id"]] for p in discovery if p["task_type"] != t]
        mean_target = torch.stack(target_vecs).mean(dim=0)
        mean_other = torch.stack(other_vecs).mean(dim=0)
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

def compute_mantel_permutation(mat1, mat2, n_permutations=10000):
    n = mat1.shape[0]
    idx = np.triu_indices(n, k=1)
    vec1 = mat1[idx]
    vec2 = mat2[idx]
    true_corr, _ = stats.spearmanr(vec1, vec2)
    
    better_count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        perm_mat2 = mat2[perm, :][:, perm]
        perm_vec2 = perm_mat2[idx]
        perm_corr, _ = stats.spearmanr(vec1, perm_vec2)
        if perm_corr >= true_corr:
            better_count += 1
            
    p_value = better_count / n_permutations
    return true_corr, p_value

def main():
    discovery = get_prompts("../../isa-head/dataset_discovery_224.json")
    
    print("Running Step 0: Extracting Deconfounded Residuals...")
    qwen_mat, task_types = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    llama_mat, _ = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    
    original_tasks = {"fact_recall", "niah", "copy", "pattern_induction", "counting"}
    orig_indices = [i for i, t in enumerate(task_types) if t in original_tasks]
    new_indices = [i for i, t in enumerate(task_types) if t not in original_tasks]
    
    # Extract Original 10 Pairs
    qwen_orig = qwen_mat[np.ix_(orig_indices, orig_indices)]
    llama_orig = llama_mat[np.ix_(orig_indices, orig_indices)]
    orig_corr, orig_p = compute_mantel_permutation(qwen_orig, llama_orig, 1000)
    
    # Extract New 18 Pairs (This actually involves correlations containing at least one new category)
    # Easiest way: extract all pairs, then filter out the original 10.
    n = len(task_types)
    idx = np.triu_indices(n, k=1)
    qwen_vec = qwen_mat[idx]
    llama_vec = llama_mat[idx]
    
    q_orig_vec, l_orig_vec = [], []
    q_new_vec, l_new_vec = [], []
    
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        t1, t2 = task_types[i], task_types[j]
        if t1 in original_tasks and t2 in original_tasks:
            q_orig_vec.append(qwen_vec[k])
            l_orig_vec.append(llama_vec[k])
        else:
            q_new_vec.append(qwen_vec[k])
            l_new_vec.append(llama_vec[k])
            
    new_corr, _ = stats.spearmanr(q_new_vec, l_new_vec) # Simple spearman for the subset
    true_corr, p_val = compute_mantel_permutation(qwen_mat, llama_mat, 10000)
    
    results = {
        "step0_deconfounded": {
            "overall_spearman_rho": true_corr,
            "mantel_permutation_p_value": p_val,
            "subgroups": {
                "original_10_pairs_rho": orig_corr,
                "new_18_pairs_rho": new_corr
            }
        }
    }
    
    os.makedirs("../outputs-isa-residual/step0", exist_ok=True)
    with open("../outputs-isa-residual/step0/step0_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Overall rho={true_corr:.4f}, p={p_val:.4f}")
    print(f"Original 10 rho={orig_corr:.4f}")
    print(f"New 18 rho={new_corr:.4f}")
    print("Saved to outputs-isa-residual/step0/step0_results.json")

if __name__ == "__main__":
    main()
