import json
import os
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import itertools
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_comparison_factorial_dataset():
    # Generate 15 prompts for each of 5 domains
    # Domain 1: Numbers
    numbers = []
    pairs1 = [(17, 12), (45, 30), (99, 14), (56, 55), (102, 8), (73, 21), (88, 87), (19, 3), (64, 46), (31, 29), (50, 40), (22, 11), (77, 33), (91, 19), (15, 10)]
    for a, b in pairs1:
        numbers.append({"prompt": f"Which is greater, {a} or {b}? \nAnswer:", "target": f" {a}", "domain": "numbers"})
        
    # Domain 2: Dates
    dates = []
    pairs2 = [("December", "January"), ("October", "March"), ("August", "May"), ("November", "February"), ("July", "June"), ("September", "April"), ("December", "July"), ("October", "August"), ("November", "September"), ("May", "January"), ("August", "March"), ("July", "February"), ("December", "October"), ("November", "May"), ("September", "June")]
    for a, b in pairs2:
        dates.append({"prompt": f"Which occurs later in the year, {a} or {b}? \nAnswer:", "target": f" {a}", "domain": "dates"})
        
    # Domain 3: Arbitrary Symbols
    symbols = []
    pairs3 = [("X", "Y"), ("Z", "W"), ("P", "Q"), ("M", "N"), ("K", "L"), ("R", "S"), ("V", "U"), ("T", "F"), ("H", "G"), ("D", "C"), ("B", "A"), ("J", "I"), ("O", "E"), ("X", "W"), ("Z", "Y")]
    for a, b in pairs3:
        symbols.append({"prompt": f"If {a} outranks {b}, which is higher? \nAnswer:", "target": f" {a}", "domain": "symbols"})
        
    # Domain 4: Lengths
    lengths = []
    pairs4 = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"), ("I", "J"), ("K", "L"), ("M", "N"), ("O", "P"), ("Q", "R"), ("S", "T"), ("U", "V"), ("W", "X"), ("Y", "Z"), ("A", "C"), ("B", "D")]
    for a, b in pairs4:
        lengths.append({"prompt": f"If {a} is taller than {b}, which is taller? \nAnswer:", "target": f" {a}", "domain": "lengths"})
        
    # Domain 5: Invented Words
    invented = []
    pairs5 = [("dax", "wug"), ("glorp", "plonk"), ("zib", "fep"), ("blick", "tulver"), ("snark", "boojum"), ("tove", "borogove"), ("mome", "rath"), ("wabe", "gyre"), ("gimble", "outgrabe"), ("frumious", "bandersnatch"), ("jubjub", "bird"), ("tumtum", "tree"), ("tulgey", "wood"), ("vorpal", "sword"), ("manxome", "foe")]
    for a, b in pairs5:
        invented.append({"prompt": f"If a {a} is heavier than a {b}, which is heavier? \nAnswer:", "target": f" {a}", "domain": "invented"})
        
    all_prompts = numbers + dates + symbols + lengths + invented
    # Just use all 75 as discovery for direction estimation
    for i, p in enumerate(all_prompts):
        p["id"] = f"comp_{i}"
        p["task_type"] = "comparison"
        
    return all_prompts

def get_base_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        return [p for p in data if p["task_type"] == "copy"]

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

def cliffs_delta(lst1, lst2):
    m, n = len(lst1), len(lst2)
    dominations = 0
    for x in lst1:
        for y in lst2:
            if x > y: dominations += 1
            elif x < y: dominations -= 1
    return dominations / (m * n)

def run_factorial_analysis(model_name, comparison_prompts, copy_prompts):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8)

    comp_res = extract_residuals(model, tokenizer, comparison_prompts, target_layer)
    copy_res = extract_residuals(model, tokenizer, copy_prompts, target_layer)
    
    # We will use the overall mean of all prompts as the baseline "other" to subtract for directions
    overall_mean = torch.cat([comp_res, copy_res]).mean(dim=0)
    
    copy_dir = copy_res.mean(dim=0) - overall_mean
    copy_dir = copy_dir / torch.norm(copy_dir)
    
    domains = ["numbers", "dates", "symbols", "lengths", "invented"]
    domain_dirs = {}
    for d in domains:
        idx = [i for i, p in enumerate(comparison_prompts) if p["domain"] == d]
        d_res = comp_res[idx]
        d_dir = d_res.mean(dim=0) - overall_mean
        domain_dirs[d] = d_dir / torch.norm(d_dir)
        
    import torch.nn.functional as F
    
    # Test 1: Invariance
    within_comp_sims = []
    comp_to_copy_sims = []
    
    for i in range(len(domains)):
        comp_to_copy_sims.append(F.cosine_similarity(domain_dirs[domains[i]].unsqueeze(0), copy_dir.unsqueeze(0)).item())
        for j in range(i+1, len(domains)):
            sim = F.cosine_similarity(domain_dirs[domains[i]].unsqueeze(0), domain_dirs[domains[j]].unsqueeze(0)).item()
            within_comp_sims.append(sim)
            
    delta = cliffs_delta(within_comp_sims, comp_to_copy_sims)
    
    # Mantel permutation for Test 1
    # We have 5 domains. We want to permute the labels of the 5 domains + 1 copy.
    # Actually, Cliff's delta is non-parametric. We can do a permutation test on the difference in means.
    true_diff = np.mean(within_comp_sims) - np.mean(comp_to_copy_sims)
    
    all_dirs = list(domain_dirs.values()) + [copy_dir]
    n_perm = 10000
    better_count = 0
    for _ in range(n_perm):
        perm_idx = np.random.permutation(len(all_dirs))
        perm_dirs = [all_dirs[idx] for idx in perm_idx]
        p_within = []
        p_cross = []
        for i in range(5):
            p_cross.append(F.cosine_similarity(perm_dirs[i].unsqueeze(0), perm_dirs[5].unsqueeze(0)).item())
            for j in range(i+1, 5):
                p_within.append(F.cosine_similarity(perm_dirs[i].unsqueeze(0), perm_dirs[j].unsqueeze(0)).item())
        if (np.mean(p_within) - np.mean(p_cross)) >= true_diff:
            better_count += 1
            
    p_val = better_count / n_perm
    
    # Test 2: Discriminability (Domain-crossed classifier)
    # Train on 4 domains, test on 1.
    discriminability_success = 0
    for holdout in domains:
        train_domains = [d for d in domains if d != holdout]
        train_res = torch.stack([domain_dirs[d] for d in train_domains])
        comp_train_dir = train_res.mean(dim=0)
        comp_train_dir = comp_train_dir / torch.norm(comp_train_dir)
        
        holdout_dir = domain_dirs[holdout]
        
        sim_to_comp = F.cosine_similarity(holdout_dir.unsqueeze(0), comp_train_dir.unsqueeze(0)).item()
        sim_to_copy = F.cosine_similarity(holdout_dir.unsqueeze(0), copy_dir.unsqueeze(0)).item()
        
        if sim_to_comp > sim_to_copy:
            discriminability_success += 1

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "within_comp_sims_mean": float(np.mean(within_comp_sims)),
        "comp_to_copy_sims_mean": float(np.mean(comp_to_copy_sims)),
        "cliffs_delta": float(delta),
        "permutation_p_value": p_val,
        "discriminability_accuracy": discriminability_success / len(domains)
    }

def main():
    print("Running Step 2: Content-Domain Factorial Control...")
    comparison_prompts = generate_comparison_factorial_dataset()
    copy_prompts = get_base_prompts("../../isa-head/dataset_discovery_224.json")
    
    qwen_results = run_factorial_analysis("Qwen/Qwen2.5-1.5B", comparison_prompts, copy_prompts)
    llama_results = run_factorial_analysis("unsloth/Llama-3.2-1B", comparison_prompts, copy_prompts)
    
    results = {
        "qwen": qwen_results,
        "llama": llama_results
    }
    
    os.makedirs("../outputs-isa-residual/step2", exist_ok=True)
    with open("../outputs-isa-residual/step2/step2_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nQwen Results:")
    print(json.dumps(qwen_results, indent=2))
    print("\nLlama Results:")
    print(json.dumps(llama_results, indent=2))

if __name__ == "__main__":
    main()
