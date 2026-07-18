import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_features_and_trajectories(model, tokenizer, prompts):
    X = torch.zeros(len(prompts), 6)
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find layers attribute in the model.")
        
    n_layers = len(layers)
    cache = {l: [] for l in range(n_layers)}
    
    def get_hook(layer_idx):
        def hook(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            if hidden.dim() == 3: val = hidden[0, -1, :].detach().cpu()
            else: val = hidden[-1, :].detach().cpu()
            cache[layer_idx].append(val)
        return hook
        
    handles = []
    for l in range(n_layers):
        handles.append(layers[l].register_forward_hook(get_hook(l)))
        
    print(f"Executing {len(prompts)} prompts across {n_layers} layers...")
    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]
        target_text = p["target"]
        
        tokens = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        prompt_len = tokens.input_ids.shape[1]
        target_len = len(target_text)
        num_digits = sum(c.isdigit() for c in prompt_text)
        density = num_digits / max(1, len(prompt_text))
        
        with torch.no_grad():
            outputs = model(**tokens)
            
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
        
    for h in handles:
        h.remove()
        
    # Standardize covariates
    for j in range(1, 6):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-8)
        
    trajectories = torch.zeros(len(prompts), n_layers, cache[0][0].shape[0])
    for l in range(n_layers):
        trajectories[:, l, :] = torch.stack(cache[l])
        
    return trajectories, X

def deconfound_layer(Y_layer, X):
    X = X.to(DEVICE)
    Y_layer = Y_layer.to(DEVICE)
    
    W = torch.linalg.lstsq(X, Y_layer).solution
    Y_pred = X @ W
    Y_pure = Y_layer - Y_pred
    
    Y_mean = Y_layer.mean(dim=0, keepdim=True)
    ss_tot = torch.sum((Y_layer - Y_mean)**2)
    ss_res = torch.sum(Y_pure**2)
    
    r2 = 1.0 - (ss_res / ss_tot).item()
    
    return Y_pure.cpu(), r2

def process_model(model_name, mapping_prompts, out_dir):
    print(f"\n{'='*50}\nProcessing {model_name}\n{'='*50}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    
    raw_trajectories, X = extract_features_and_trajectories(model, tokenizer, mapping_prompts)
    N, L, D = raw_trajectories.shape
    
    deconfounded_trajectories = torch.zeros_like(raw_trajectories)
    r2_values = []
    
    print("Executing independent per-layer 5-covariate deconfounding...")
    for l in range(L):
        Y_pure, r2 = deconfound_layer(raw_trajectories[:, l, :], X)
        deconfounded_trajectories[:, l, :] = Y_pure
        r2_values.append(r2)
        
    print(f"Layer R^2 stats: Min={min(r2_values):.4f}, Max={max(r2_values):.4f}, Mean={sum(r2_values)/len(r2_values):.4f}")
    
    categories = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    centroids = torch.zeros(6, L, D)
    normalized_centroids = torch.zeros(6, L, D)
    
    print("Computing L2-normalized trajectory centroids for DTW mapping...")
    for c_idx, cat in enumerate(categories):
        cat_indices = [i for i, p in enumerate(mapping_prompts) if p["task_type"] == cat]
        
        for l in range(L):
            mean_vec = deconfounded_trajectories[cat_indices, l, :].mean(dim=0)
            centroids[c_idx, l, :] = mean_vec
            normalized_centroids[c_idx, l, :] = mean_vec / (mean_vec.norm() + 1e-8)
            
    m_name = model_name.split("/")[-1]
    m_dir = os.path.join(out_dir, m_name)
    os.makedirs(m_dir, exist_ok=True)
    
    torch.save(raw_trajectories, os.path.join(m_dir, "raw_trajectories.pt"))
    torch.save(deconfounded_trajectories, os.path.join(m_dir, "deconfounded_trajectories.pt"))
    torch.save(centroids, os.path.join(m_dir, "centroids.pt"))
    torch.save(normalized_centroids, os.path.join(m_dir, "normalized_centroids.pt"))
    
    with open(os.path.join(m_dir, "r2_values.json"), "w") as f:
        json.dump(r2_values, f, indent=2)
        
    print(f"Successfully saved all tensors and stats to {m_dir}")
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    print("Starting Trajectory Extraction Pipeline...")
    
    with open("../outputs/dataset/trajectory_mapping.json", "r", encoding="utf-8") as f:
        mapping_prompts = json.load(f)
        
    out_dir = "../outputs/trajectories"
    
    models = [
        "Qwen/Qwen2.5-1.5B",
        "unsloth/Llama-3.2-1B",
        "microsoft/phi-1_5"
    ]
    
    for m in models:
        process_model(m, mapping_prompts, out_dir)
        
    print("\nExtraction Complete. All data perfectly formatted for Section 2 (DTW Mapping).")

if __name__ == "__main__":
    main()
