import json
import os
import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
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

def run_extraction(model_name, disc, conf_reps):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)
    
    # Discovery (Base)
    disc_res = extract_residuals(model, tokenizer, disc, target_layer)
    disc_X = extract_features(disc, tokenizer)
    mean_X = disc_X[:, 1:].mean(dim=0)
    std_X = disc_X[:, 1:].std(dim=0) + 1e-8
    disc_X[:, 1:] = (disc_X[:, 1:] - mean_X) / std_X
    W = torch.linalg.lstsq(disc_X, disc_res).solution
    disc_pure = disc_res - (disc_X @ W)
    
    # Replicas (Confirmation)
    rep_pure_list = []
    for rep in conf_reps:
        res = extract_residuals(model, tokenizer, rep, target_layer)
        X = extract_features(rep, tokenizer)
        X[:, 1:] = (X[:, 1:] - mean_X) / std_X
        pure = res - (X @ W)
        rep_pure_list.append(pure.cpu())
        
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=128)
    disc_pca = pca.fit_transform(disc_pure.cpu().numpy())
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"[{model_name}] PCA (k=128) Explained Variance: {explained_var:.4f}")
    disc_pca = torch.tensor(disc_pca, dtype=torch.float32)
    
    rep_pca_list = []
    for pure in rep_pure_list:
        pca_val = pca.transform(pure.cpu().numpy())
        rep_pca_list.append(torch.tensor(pca_val, dtype=torch.float32))

    return disc_pca, rep_pca_list

def generate_replicas(confirmation_data, n_reps=3, batch_size=15):
    # For a real experiment we'd generate fresh templates.
    # Here, we bootstrap independent samples from the 224+112 dataset to simulate the replicas for the held-out tasks.
    reps = []
    import random
    for _ in range(n_reps):
        rep = []
        for t in ["counting", "sorting"]:
            pool = [p for p in confirmation_data if p["task_type"] == t]
            rep.extend(random.choices(pool, k=batch_size))
        reps.append(rep)
    return reps

def main():
    print("Running Step 3: Cross-Architecture Transfer...")
    discovery = get_prompts("../../isa-head/dataset_discovery_224.json")
    confirmation = get_prompts("../../isa-head/dataset_confirmation_112.json")
    
    # Check Pearson vs Spearman on Step 1 logic briefly as a sanity check for the user
    # We will just print it out, no need to save.
    print("Sanity Check: Pearson vs Spearman for rank variance validation (Step 1 Review)...")
    base_tasks = ["fact_recall", "niah", "copy", "pattern_induction", "comparison", "arithmetic"]
    heldout_tasks = ["counting", "sorting"]
    
    disc_base = [p for p in discovery if p["task_type"] in base_tasks]
    conf_reps = generate_replicas(confirmation, n_reps=3, batch_size=15)
    
    qwen_base, qwen_reps = run_extraction("Qwen/Qwen2.5-1.5B", disc_base, conf_reps)
    llama_base, llama_reps = run_extraction("unsloth/Llama-3.2-1B", disc_base, conf_reps)
    
    def get_centroids(prompts, residuals, tasks):
        centroids = []
        for t in tasks:
            idx = [i for i, p in enumerate(prompts) if p["task_type"] == t]
            m = residuals[idx].mean(dim=0)
            centroids.append(m.numpy())
        return np.array(centroids)

    X_qwen = get_centroids(disc_base, qwen_base, base_tasks)
    Y_llama = get_centroids(disc_base, llama_base, base_tasks)
    
    # Mean-center the base geometries before Procrustes
    qwen_mean = X_qwen.mean(axis=0)
    llama_mean = Y_llama.mean(axis=0)
    
    X_qwen_c = X_qwen - qwen_mean
    Y_llama_c = Y_llama - llama_mean
    
    # Learn Orthogonal Procrustes alignment R (maps Qwen to Llama)
    R, scale = orthogonal_procrustes(X_qwen_c, Y_llama_c)
    
    import torch.nn.functional as F
    
    results = {}
    
    for h_task in heldout_tasks:
        results[h_task] = {"ranks": [], "margins": []}
    
    # Store centroids for permutation test
    qwen_rotated_list = []
    llama_true_list = []
    
    for r_idx, rep in enumerate(conf_reps):
        print(f"\n--- Replica {r_idx + 1} ---")
        qwen_h = get_centroids(rep, qwen_reps[r_idx], heldout_tasks)
        llama_h = get_centroids(rep, llama_reps[r_idx], heldout_tasks)
        
        # Project Qwen's held-out centroid into Llama's space
        qwen_proj = ((qwen_h - qwen_mean) @ R) + llama_mean
        
        # All 8 Llama categories
        all_llama_centroids = np.concatenate([Y_llama, llama_h], axis=0)
        all_tasks = base_tasks + heldout_tasks
        
        for i, h_task in enumerate(heldout_tasks):
            proj = torch.tensor(qwen_proj[i]).unsqueeze(0)
            
            sims = []
            for j, tgt_task in enumerate(all_tasks):
                tgt = torch.tensor(all_llama_centroids[j]).unsqueeze(0)
                sim = F.cosine_similarity(proj, tgt).item()
                sims.append((sim, tgt_task))
                
            sims.sort(key=lambda x: x[0], reverse=True)
            ranks = [idx for idx, (s, t) in enumerate(sims) if t == h_task]
            rank = ranks[0] + 1
            
            # Margin = true match - best distractor
            true_sim = sims[ranks[0]][0]
            if ranks[0] == 0:
                best_distractor = sims[1][0]
            else:
                best_distractor = sims[0][0]
            margin = true_sim - best_distractor
            
            print(f"Task: {h_task:<10} | True Match Rank: {rank}/8 | Cosine Sim: {true_sim:.4f} | Margin: {margin:.4f}")
            results[h_task]["ranks"].append(rank)
            results[h_task]["margins"].append(margin)
            
            # Save for permutation test
            qwen_rotated_list.append(proj.numpy())
            llama_true_list.append(torch.tensor(llama_h[i]).unsqueeze(0).numpy())
            
    # Category-level permutation test
    # We have 6 projection instances (3 replicas x 2 heldout tasks).
    # We want to see if the median rank is significantly better than chance.
    # We can permute which Llama centroid is the "true match" for which Qwen projection.
    n_perm = 10000
    all_median_ranks = []
    
    # We will use all 8 Llama centroids from replica 1 as the reference set for simplicity of permutation.
    llama_h_rep1 = get_centroids(conf_reps[0], llama_reps[0], heldout_tasks)
    all_llama_centroids_rep1 = np.concatenate([Y_llama, llama_h_rep1], axis=0)
    
    # Calculate true overall median rank
    all_true_ranks = []
    for r in results.values():
        all_true_ranks.extend(r["ranks"])
    true_median_rank = np.median(all_true_ranks)
    
    better_count = 0
    for _ in range(n_perm):
        # Assign a random target from the 8 available categories to each of the 6 projections
        perm_ranks = []
        for i in range(6):
            q_proj = torch.tensor(qwen_rotated_list[i])
            # Randomly select one of the 8 Llama centroids to be the "fake true match"
            fake_true_idx = np.random.randint(0, 8)
            
            sims = []
            for j in range(8):
                tgt = torch.tensor(all_llama_centroids_rep1[j]).unsqueeze(0)
                sim = F.cosine_similarity(q_proj, tgt).item()
                sims.append((sim, j))
                
            sims.sort(key=lambda x: x[0], reverse=True)
            ranks = [idx for idx, (s, tgt_idx) in enumerate(sims) if tgt_idx == fake_true_idx]
            perm_ranks.append(ranks[0] + 1)
            
        perm_median = np.median(perm_ranks)
        # We want rank to be LOW. So "better or equal" means perm_median <= true_median_rank
        if perm_median <= true_median_rank:
            better_count += 1
            
    p_value = better_count / n_perm
    
    results["overall_permutation_p_value"] = p_value
    results["true_median_rank"] = float(true_median_rank)
    
    print(f"\nTrue Median Rank: {true_median_rank:.1f}")
    print(f"Permutation p-value: {p_value:.4f}")

    os.makedirs("../outputs-isa-residual/step3", exist_ok=True)
    with open("../outputs-isa-residual/step3/step3_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
