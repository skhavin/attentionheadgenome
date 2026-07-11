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
    for t in ["comparison", "copy", "fact_recall"]:
        t_idx = [i for i, p in enumerate(discovery) if p["task_type"] == t]
        o_idx = [i for i, p in enumerate(discovery) if p["task_type"] != t]
        d = Y_pure[t_idx].mean(dim=0) - Y_pure[o_idx].mean(dim=0)
        dirs[t] = d / torch.norm(d)
        
    return dirs

def substitution_patch(model, tokenizer, prompt, target_layer, ablate_dir, inject_dir, alpha, comp_vocab):
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    def patch_hook(m, a, o):
        hidden = o[0] if isinstance(o, tuple) else o
        a_dir = ablate_dir.to(hidden.dtype)
        i_dir = inject_dir.to(hidden.dtype)
        
        if hidden.dim() == 3: 
            # Erase specific direction
            proj = (hidden[:, -1, :] @ a_dir).unsqueeze(-1) * a_dir
            hidden[:, -1, :] = hidden[:, -1, :] - proj
            # Inject new direction
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * i_dir
        else:
            proj = (hidden[-1, :] @ a_dir) * a_dir
            hidden[-1, :] = hidden[-1, :] - proj
            hidden[-1, :] = hidden[-1, :] + alpha * i_dir
        return (hidden,) if isinstance(o, tuple) else hidden
            
    handle = model.model.layers[target_layer].register_forward_hook(patch_hook)
        
    with torch.no_grad():
        outputs = model(**tokens)
        
    handle.remove()
        
    logits = outputs.logits[0, -1, :].float()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs, 200)
    comp_prob_mass = 0.0
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode(idx).strip().lower()
        if token_str in comp_vocab:
            comp_prob_mass += prob.item()
            
    return comp_prob_mass

def main():
    print("Running Representation Substitution Causal Test...")
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
    
    dirs = extract_pure_directions(model, tokenizer, discovery, int(n_layers * 0.8))
    copy_dir = dirs["copy"]
    comp_dir = dirs["comparison"]
    fact_dir = dirs["fact_recall"]
    
    torch.manual_seed(42)
    random_dir = torch.randn_like(comp_dir)
    random_dir = random_dir / torch.norm(random_dir)
    
    copy_prompts = [p["prompt"] for p in discovery if p["task_type"] == "copy"][:20]
    comp_vocab = {">", "<", "true", "false", "yes", "no", "equal", "greater", "less"}
    
    alpha = 50.0 # Increase alpha to 50 for substitution to ensure maximum signal injection
    
    print("--- Representation Substitution on Copy Prompts ---")
    print("Baseline Comparison Mass (No patch): 0.0020")
    
    for l in layers_to_sweep:
        print(f"\nLayer {l} ({(l/n_layers)*100:.0f}% depth):")
        
        sub_true = np.mean([substitution_patch(model, tokenizer, p, l, copy_dir, comp_dir, alpha, comp_vocab) for p in copy_prompts])
        sub_fact = np.mean([substitution_patch(model, tokenizer, p, l, copy_dir, fact_dir, alpha, comp_vocab) for p in copy_prompts])
        sub_rand = np.mean([substitution_patch(model, tokenizer, p, l, copy_dir, random_dir, alpha, comp_vocab) for p in copy_prompts])
        sub_wrong = np.mean([substitution_patch(model, tokenizer, p, l, fact_dir, comp_dir, alpha, comp_vocab) for p in copy_prompts])
        
        print(f"  Substitute (Erase Copy, Add Comp):            {sub_true:.4f}")
        print(f"  Control (Erase Copy, Add Fact_Recall):        {sub_fact:.4f}")
        print(f"  Control (Erase Copy, Add Random):             {sub_rand:.4f}")
        print(f"  Directionality (Erase Fact_Recall, Add Comp): {sub_wrong:.4f}")

if __name__ == "__main__":
    main()
