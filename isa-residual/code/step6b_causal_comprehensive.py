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
        
        X[i, 0] = 1.0
        X[i, 1] = float(prompt_len)
        X[i, 2] = float(target_len)
        X[i, 3] = float(density)
        
    for j in range(1, 4):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-8)
        
    X = X.to(DEVICE)
    Y = torch.stack(residuals).to(DEVICE)
    
    W = torch.linalg.lstsq(X, Y).solution
    Y_pure = Y - (X @ W)
    
    dirs = {}
    for t in ["comparison", "fact_recall"]:
        t_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        o_idx = [i for i, p in enumerate(discovery) if p["task_type"] != t]
        d = Y_pure[t_idx].mean(dim=0) - Y_pure[o_idx].mean(dim=0)
        dirs[t] = d / torch.norm(d)
        
    return dirs

def steering_patch(model, tokenizer, prompt, target_layer, intervention_dir, alpha, comp_vocab):
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    if intervention_dir is not None:
        def patch_hook(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            if hidden.dim() == 3: 
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * intervention_dir.to(hidden.dtype)
            else:
                hidden[-1, :] = hidden[-1, :] + alpha * intervention_dir.to(hidden.dtype)
            return (hidden,) if isinstance(o, tuple) else hidden
            
        handle = model.model.layers[target_layer].register_forward_hook(patch_hook)
    else:
        handle = None
        
    with torch.no_grad():
        outputs = model(**tokens)
        
    if handle: handle.remove()
        
    logits = outputs.logits[0, -1, :].float()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs, 200)
    comp_prob_mass = 0.0
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode(idx).strip().lower()
        if token_str in comp_vocab:
            comp_prob_mass += prob.item()
            
    return comp_prob_mass

def ablation_patch(model, tokenizer, prompt_obj, target_layer, ablate_dir):
    tokens = tokenizer(prompt_obj["prompt"], return_tensors="pt").to(DEVICE)
    
    # We want the probability of the correct target token
    target_token_str = prompt_obj["target"].strip()
    target_token_id = tokenizer.encode(" " + target_token_str, add_special_tokens=False)[-1]
    # some models encode differently, let's just get the top 500 and sum if string matches
    
    if ablate_dir is not None:
        def patch_hook(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            dir_dt = ablate_dir.to(hidden.dtype)
            if hidden.dim() == 3: 
                # Orthogonal projection out
                proj = (hidden[:, -1, :] @ dir_dt).unsqueeze(-1) * dir_dt
                hidden[:, -1, :] = hidden[:, -1, :] - proj
            else:
                proj = (hidden[-1, :] @ dir_dt) * dir_dt
                hidden[-1, :] = hidden[-1, :] - proj
            return (hidden,) if isinstance(o, tuple) else hidden
            
        handle = model.model.layers[target_layer].register_forward_hook(patch_hook)
    else:
        handle = None
        
    with torch.no_grad():
        outputs = model(**tokens)
        
    if handle: handle.remove()
        
    logits = outputs.logits[0, -1, :].float()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Find probability of target
    top_probs, top_indices = torch.topk(probs, 200)
    target_prob_mass = 0.0
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode(idx).strip().lower()
        if token_str == target_token_str.lower():
            target_prob_mass += prob.item()
            
    return target_prob_mass

def main():
    print("Running Comprehensive Causal Sweep (Sufficiency & Necessity)...")
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers_to_sweep = [
        int(n_layers * 0.2),
        int(n_layers * 0.4),
        int(n_layers * 0.6),
        int(n_layers * 0.8)
    ]
    
    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    
    # Extract directions (using 80% layer as the "read-out" vector)
    dirs = extract_pure_directions(model, tokenizer, discovery, int(n_layers * 0.8))
    comp_dir = dirs["comparison"]
    control_dir = dirs["fact_recall"]
    
    torch.manual_seed(42)
    random_dir = torch.randn_like(comp_dir)
    random_dir = random_dir / torch.norm(random_dir)
    
    print("\n--- Phase 1: Sufficiency (Layer Sweep Steering) ---")
    copy_prompts = [p["prompt"] for p in discovery if p["task_type"] == "copy"][:20]
    comp_vocab = {">", "<", "true", "false", "yes", "no", "equal", "greater", "less"}
    
    alpha = 30.0
    
    mass_base = np.mean([steering_patch(model, tokenizer, p, layers_to_sweep[0], None, 0.0, comp_vocab) for p in copy_prompts])
    print(f"Baseline Comparison Mass (No patch): {mass_base:.4f}")
    
    for l in layers_to_sweep:
        print(f"\nLayer {l} ({(l/n_layers)*100:.0f}% depth):")
        comp_mass = np.mean([steering_patch(model, tokenizer, p, l, comp_dir, alpha, comp_vocab) for p in copy_prompts])
        fact_mass = np.mean([steering_patch(model, tokenizer, p, l, control_dir, alpha, comp_vocab) for p in copy_prompts])
        rand_mass = np.mean([steering_patch(model, tokenizer, p, l, random_dir, alpha, comp_vocab) for p in copy_prompts])
        
        print(f"  Comparison Patch: {comp_mass:.4f}")
        print(f"  Fact_Recall Patch: {fact_mass:.4f}")
        print(f"  Random Patch:      {rand_mass:.4f}")
        
    print("\n--- Phase 2: Necessity (Orthogonal Projection Ablation) ---")
    # Use fact_recall prompts because Qwen 1.5B actually solves them zero-shot (baseline prob ~0.90)
    fr_prompts_obj = [p for p in discovery if p["task_type"] == "fact_recall"][:20]
    
    target_base = np.mean([ablation_patch(model, tokenizer, p, layers_to_sweep[0], None) for p in fr_prompts_obj])
    print(f"Baseline True Target Probability (No ablation): {target_base:.4f}")
    
    for l in layers_to_sweep:
        print(f"\nLayer {l} ({(l/n_layers)*100:.0f}% depth):")
        fact_drop = np.mean([ablation_patch(model, tokenizer, p, l, control_dir) for p in fr_prompts_obj])
        comp_drop = np.mean([ablation_patch(model, tokenizer, p, l, comp_dir) for p in fr_prompts_obj])
        rand_drop = np.mean([ablation_patch(model, tokenizer, p, l, random_dir) for p in fr_prompts_obj])
        
        print(f"  Ablate Fact_Recall (True Opcode): {fact_drop:.4f}")
        print(f"  Ablate Comparison (Control):      {comp_drop:.4f}")
        print(f"  Ablate Random (Control):          {rand_drop:.4f}")

if __name__ == "__main__":
    main()
