import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def run_model_analysis(model_name, discovery, confirmation):
    print(f"\n{'='*50}\nEvaluating Model: {model_name}\n{'='*50}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    n_layers = model.config.num_hidden_layers
    # Dynamically target the layer where "Answer" typically spikes (~80% depth)
    target_layer = int(n_layers * 0.8)
    print(f"Targeting Layer {target_layer} for Residual extraction.")

    # Dictionary to store residual vectors: prompt_id -> vector
    def extract_residuals(prompts, desc):
        residuals = {}
        
        for item in tqdm(prompts, desc=desc):
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
            residuals[item["id"]] = cache["val"].cpu().float()
            
        return residuals

    disc_res = extract_residuals(discovery, "Discovery Set")
    conf_res = extract_residuals(confirmation, "Confirmation Set")
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Compute Discovery Directions
    task_types = list(set([p["task_type"] for p in discovery]))
    directions = {}
    
    for t in task_types:
        target_vecs = [disc_res[p["id"]] for p in discovery if p["task_type"] == t]
        other_vecs = [disc_res[p["id"]] for p in discovery if p["task_type"] != t]
        
        mean_target = torch.stack(target_vecs).mean(dim=0)
        mean_other = torch.stack(other_vecs).mean(dim=0)
        
        direction = mean_target - mean_other
        direction = direction / torch.norm(direction)
        directions[t] = direction
        
    # Step 3: Within-Model Confirmation (Projection)
    print("\n--- Within-Model Confirmation (Mann-Whitney U) ---")
    all_passed = True
    for t in task_types:
        # Same-type scores
        same_type_prompts = [p for p in confirmation if p["task_type"] == t]
        same_scores = []
        for p in same_type_prompts:
            vec = conf_res[p["id"]]
            score = torch.dot(vec, directions[t]).item()
            same_scores.append(score)
            
        # Different-type scores (Control)
        diff_type_prompts = [p for p in confirmation if p["task_type"] != t]
        diff_scores = []
        for p in diff_type_prompts:
            vec = conf_res[p["id"]]
            score = torch.dot(vec, directions[t]).item()
            diff_scores.append(score)
            
        try:
            stat, p_val = stats.mannwhitneyu(same_scores, diff_scores, alternative='greater')
        except:
            p_val = 1.0
            
        print(f"Task: {t:<18} | MWU p-value: {p_val:.4e}")
        if p_val >= 0.05:
            all_passed = False
            
    if not all_passed:
        print(">> WARNING: Within-model confirmation failed for one or more tasks (p >= 0.05).")
    else:
        print(">> SUCCESS: Within-model structural signatures hold for all tasks.")
        
    # Build Representational Similarity Matrix (RSM)
    task_types.sort() # Ensure consistent ordering
    n_tasks = len(task_types)
    rsm = np.zeros((n_tasks, n_tasks))
    
    import torch.nn.functional as F
    for i, t1 in enumerate(task_types):
        for j, t2 in enumerate(task_types):
            sim = F.cosine_similarity(directions[t1].unsqueeze(0), directions[t2].unsqueeze(0)).item()
            rsm[i, j] = sim
            
    # Extract upper triangle (excluding diagonal)
    upper_tri = []
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            upper_tri.append(rsm[i, j])
            
    return np.array(upper_tri)

def main():
    discovery = get_prompts("dataset_discovery_224.json")
    confirmation = get_prompts("dataset_confirmation_112.json")
    
    print(f"Loaded Discovery Set: {len(discovery)} prompts")
    print(f"Loaded Confirmation Set: {len(confirmation)} prompts")
    
    qwen_rsm = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery, confirmation)
    llama_rsm = run_model_analysis("unsloth/Llama-3.2-1B", discovery, confirmation)
    
    if qwen_rsm is not None and llama_rsm is not None:
        print(f"\n{'='*50}\nStep 4: Cross-Architecture RSA Correlation\n{'='*50}")
        # Spearman correlation between Qwen's RSM upper triangle and Llama's RSM upper triangle
        corr, p_val = stats.spearmanr(qwen_rsm, llama_rsm)
        
        print(f"RSA Spearman Correlation (rho): {corr:.4f}")
        print(f"Permutation p-value (approx):   {p_val:.4e}")
        
        if corr > 0 and p_val < 0.05:
            print("\n>> FINDING HOLDS: Significant cross-architecture similarity!")
            print(">> The abstract relational structure of computation types transfers across architectures.")
        else:
            print("\n>> FINDING FAILS: No significant positive correlation found.")
            print(">> Falsification Condition met. The 'Illusion of Mechanism' stands.")

if __name__ == "__main__":
    main()
