import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def run_model_analysis(model_name, discovery):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return []

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    residuals = {}
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
        residuals[item["id"]] = cache["val"].cpu().float()

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    task_types = list(set([p["task_type"] for p in discovery]))
    task_types.sort()
    directions = {}
    for t in task_types:
        target_vecs = [residuals[p["id"]] for p in discovery if p["task_type"] == t]
        other_vecs = [residuals[p["id"]] for p in discovery if p["task_type"] != t]
        mean_target = torch.stack(target_vecs).mean(dim=0)
        mean_other = torch.stack(other_vecs).mean(dim=0)
        direction = mean_target - mean_other
        directions[t] = direction / torch.norm(direction)
        
    pairs = []
    import torch.nn.functional as F
    n_tasks = len(task_types)
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            t1, t2 = task_types[i], task_types[j]
            sim = F.cosine_similarity(directions[t1].unsqueeze(0), directions[t2].unsqueeze(0)).item()
            pairs.append((t1, t2, sim))
    return pairs

def main():
    discovery = get_prompts("dataset_discovery_224.json")
    
    qwen_pairs = run_model_analysis("Qwen/Qwen2.5-1.5B", discovery)
    llama_pairs = run_model_analysis("unsloth/Llama-3.2-1B", discovery)
    
    original_tasks = {"fact_recall", "niah", "copy", "pattern_induction", "counting"}
    
    print("\n--- Full 28 Pairwise Similarities ---")
    print(f"{'Pair':<35} | {'Qwen Sim':<10} | {'Llama Sim':<10} | {'Group'}")
    print("-" * 75)
    
    qwen_orig, llama_orig = [], []
    qwen_new, llama_new = [], []
    
    for (t1, t2, q_sim), (_, _, l_sim) in zip(qwen_pairs, llama_pairs):
        is_original = (t1 in original_tasks) and (t2 in original_tasks)
        group = "Original 10" if is_original else "New 18"
        print(f"{t1}-{t2:<20} | {q_sim:10.4f} | {l_sim:10.4f} | {group}")
        
        if is_original:
            qwen_orig.append(q_sim)
            llama_orig.append(l_sim)
        else:
            qwen_new.append(q_sim)
            llama_new.append(l_sim)
            
    print("\n--- Sub-Group Spearman Correlations ---")
    orig_corr, orig_p = stats.spearmanr(qwen_orig, llama_orig)
    print(f"Original 10 Pairs Correlation: {orig_corr:.4f} (p={orig_p:.4f})")
    
    new_corr, new_p = stats.spearmanr(qwen_new, llama_new)
    print(f"New 18 Pairs Correlation:      {new_corr:.4f} (p={new_p:.4f})")
    
    overall_corr, overall_p = stats.spearmanr(qwen_orig + qwen_new, llama_orig + llama_new)
    print(f"Overall 28 Pairs Correlation:  {overall_corr:.4f} (p={overall_p:.4e})")

if __name__ == "__main__":
    main()
