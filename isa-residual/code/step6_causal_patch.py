import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_pure_directions(model, tokenizer, discovery, target_layer):
    X = torch.zeros(len(discovery), 4)
    residuals = []
    
    for i, p in enumerate(discovery):
        tokens = tokenizer(p["prompt"], return_tensors="pt").to(DEVICE)
        prompt_len = tokens.input_ids.shape[1]
        target_len = len(p["target"])
        num_digits = sum(c.isdigit() for c in p["prompt"])
        density = num_digits / max(1, len(p["prompt"]))
        
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
        
        X[i, 0] = 1.0 # Bias
        X[i, 1] = float(prompt_len)
        X[i, 2] = float(target_len)
        X[i, 3] = float(density)
        
    for j in range(1, 4):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-8)
        
    X = X.to(DEVICE)
    Y = torch.stack(residuals).to(DEVICE)
    
    W = torch.linalg.lstsq(X, Y).solution
    Y_pure = Y - (X @ W)
    
    # Calculate pure centroids
    dirs = {}
    for t in ["comparison", "fact_recall"]:
        t_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        o_idx = [i for i, p in enumerate(discovery) if p["task_type"] != t]
        d = Y_pure[t_idx].mean(dim=0) - Y_pure[o_idx].mean(dim=0)
        dirs[t] = d / torch.norm(d)
        
    return dirs

def patch_and_evaluate(model, tokenizer, prompt, target_layer, intervention_dir, alpha, comp_vocab):
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    if intervention_dir is not None:
        def patch_hook(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            if hidden.dim() == 3: 
                # Add to the last sequence position
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * intervention_dir.to(hidden.dtype)
            else:
                hidden[-1, :] = hidden[-1, :] + alpha * intervention_dir.to(hidden.dtype)
            return (hidden,) if isinstance(o, tuple) else hidden
            
        handle = model.model.layers[target_layer].register_forward_hook(patch_hook)
    else:
        handle = None
        
    with torch.no_grad():
        outputs = model(**tokens)
        
    if handle:
        handle.remove()
        
    logits = outputs.logits[0, -1, :].float()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get top 200 tokens
    top_probs, top_indices = torch.topk(probs, 200)
    
    comp_prob_mass = 0.0
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode(idx).strip().lower()
        if token_str in comp_vocab:
            comp_prob_mass += prob.item()
            
    return comp_prob_mass

def main():
    print("Running Causal Patching Experiment...")
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.8) # 80% depth (same as extraction)
    
    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    
    print("Extracting pure structural directions...")
    dirs = extract_pure_directions(model, tokenizer, discovery, target_layer)
    comp_dir = dirs["comparison"]
    control_dir = dirs["fact_recall"]
    
    # Matched-norm random vector
    torch.manual_seed(42)
    random_dir = torch.randn_like(comp_dir)
    random_dir = random_dir / torch.norm(random_dir)
    
    # We will test on 20 Copy prompts
    copy_prompts = [p["prompt"] for p in discovery if p["task_type"] == "copy"][:20]
    
    comp_vocab = {">", "<", "true", "false", "yes", "no", "equal", "greater", "less"}
    
    alphas = [0.0, 10.0, 30.0, 50.0]
    
    results = {}
    
    for alpha in alphas:
        if alpha == 0.0:
            print("\nEvaluating Baseline (alpha=0.0)")
            mass = np.mean([patch_and_evaluate(model, tokenizer, p, target_layer, None, 0.0, comp_vocab) for p in copy_prompts])
            results["baseline"] = mass
            print(f"Baseline Comparison Mass: {mass:.4f}")
            continue
            
        print(f"\nEvaluating Interventions at alpha={alpha}...")
        
        comp_mass = np.mean([patch_and_evaluate(model, tokenizer, p, target_layer, comp_dir, alpha, comp_vocab) for p in copy_prompts])
        control_mass = np.mean([patch_and_evaluate(model, tokenizer, p, target_layer, control_dir, alpha, comp_vocab) for p in copy_prompts])
        rand_mass = np.mean([patch_and_evaluate(model, tokenizer, p, target_layer, random_dir, alpha, comp_vocab) for p in copy_prompts])
        
        results[f"alpha_{alpha}"] = {
            "comparison_patch": comp_mass,
            "fact_recall_patch": control_mass,
            "random_patch": rand_mass
        }
        
        print(f"  Comparison Patch Mass: {comp_mass:.4f}")
        print(f"  Fact_Recall Patch Mass: {control_mass:.4f}")
        print(f"  Random Patch Mass:      {rand_mass:.4f}")
        
    os.makedirs("../outputs-isa-residual/step6_causal", exist_ok=True)
    with open("../outputs-isa-residual/step6_causal/patching_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
