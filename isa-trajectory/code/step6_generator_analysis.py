import json
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        val_prompts = json.load(f)
        
    categories_all = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    cat_indices = {c: [] for c in categories_all}
    for i, p in enumerate(val_prompts):
        cat_indices[p["task_type"]].append(i)

    # 1. Define Trajectory Direction
    raw_T = torch.load("../outputs/trajectories/Qwen2.5-1.5B/val_raw_trajectories.pt", map_location="cpu")
    
    src_cat, tgt_cat = "arithmetic", "sorting"
    src_idx = cat_indices[src_cat]
    tgt_idx = cat_indices[tgt_cat]
    
    src_centroids = raw_T[src_idx].mean(dim=0).to(model.device) # [num_layers, hidden_size]
    tgt_centroids = raw_T[tgt_idx].mean(dim=0).to(model.device)
    
    trajectory_dirs = tgt_centroids - src_centroids
    # Normalize trajectory directions
    norms = torch.norm(trajectory_dirs, dim=-1, keepdim=True)
    trajectory_dirs = trajectory_dirs / (norms + 1e-8) # [num_layers, hidden_size]
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # 2. Setup Hooks for DTA
    head_outputs = {l: [] for l in range(num_layers)}
    mlp_outputs = {l: [] for l in range(num_layers)}
    residuals = {l: [] for l in range(num_layers + 1)}
    
    def get_residual_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                residuals[layer_idx + 1].append(output[0][:, -1, :].detach().cpu())
            else:
                residuals[layer_idx + 1].append(output[:, -1, :].detach().cpu())
        return hook
        
    def get_o_proj_hook(layer_idx):
        def hook(module, input, output):
            # input to o_proj is a tuple (hidden_states,)
            z = input[0][:, -1, :] # [batch, hidden_size]
            W_O = module.weight # [hidden_size, hidden_size]
            
            heads_out = []
            for h in range(num_heads):
                z_h = z[:, h*head_dim : (h+1)*head_dim]
                W_O_h = W_O[:, h*head_dim : (h+1)*head_dim]
                head_out = torch.matmul(z_h, W_O_h.T) # [batch, hidden_size]
                heads_out.append(head_out.detach().cpu())
            
            # heads_out is a list of [batch, hidden_size]
            head_outputs[layer_idx].append(torch.stack(heads_out, dim=1)) # [batch, num_heads, hidden_size]
        return hook
        
    def get_mlp_hook(layer_idx):
        def hook(module, input, output):
            mlp_outputs[layer_idx].append(output[:, -1, :].detach().cpu())
        return hook

    handles = []
    # Hook embedding to get layer 0 input
    def emb_hook(module, input, output):
        residuals[0].append(output[:, -1, :].detach().cpu())
    handles.append(model.model.embed_tokens.register_forward_hook(emb_hook))
    
    for l in range(num_layers):
        layer_module = model.model.layers[l]
        handles.append(layer_module.register_forward_hook(get_residual_hook(l)))
        handles.append(layer_module.self_attn.o_proj.register_forward_hook(get_o_proj_hook(l)))
        handles.append(layer_module.mlp.register_forward_hook(get_mlp_hook(l)))
        
    # Run forward passes on Arithmetic prompts
    prompts = [val_prompts[i]["prompt"] for i in src_idx]
    batch_size = 4
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            model(**inputs)
            
    for h in handles:
        h.remove()
        
    # Aggregate lists
    for l in range(num_layers):
        head_outputs[l] = torch.cat(head_outputs[l], dim=0).to(model.device) # [N, num_heads, hidden_size]
        mlp_outputs[l] = torch.cat(mlp_outputs[l], dim=0).to(model.device)   # [N, hidden_size]
    for l in range(num_layers + 1):
        residuals[l] = torch.cat(residuals[l], dim=0).to(model.device)       # [N, hidden_size]
        
    # 3. Compute Dual-Metric DTA
    dta_static_heads = np.zeros((num_layers, num_heads))
    dta_static_mlps = np.zeros(num_layers)
    dta_trans_heads = np.zeros((num_layers, num_heads))
    dta_trans_mlps = np.zeros(num_layers)
    
    # We want to trace the trajectory specifically during the "Rise Phase" (Layers 10-20)
    # But we'll compute it for all layers.
    
    for l in range(num_layers):
        target_dir = trajectory_dirs[l] # [hidden_size]
        
        # Static DTA: Mean across prompts of (Component * TargetDir)
        # Heads: [N, num_heads, hidden_size] * [hidden_size] -> [N, num_heads] -> mean [num_heads]
        static_h = (head_outputs[l] * target_dir).sum(dim=-1).mean(dim=0)
        dta_static_heads[l] = static_h.cpu().numpy()
        
        # MLP: [N, hidden_size] * [hidden_size]
        static_m = (mlp_outputs[l] * target_dir).sum(dim=-1).mean(dim=0)
        dta_static_mlps[l] = static_m.cpu().numpy()
        
        # Transition DTA: We project the layer-to-layer residual delta (r_{l+1} - r_l) onto TargetDir.
        # By linearity of the residual stream: r_{l+1} - r_l = sum(Heads) + MLP.
        delta_r = residuals[l+1] - residuals[l] # [N, hidden_size]
        total_transition = (delta_r * target_dir).sum(dim=-1).mean(dim=0) # Scalar
        
        if abs(total_transition.item()) > 1e-6:
            dta_trans_heads[l] = (static_h / total_transition).cpu().numpy()
            dta_trans_mlps[l] = (static_m / total_transition).cpu().numpy()
        else:
            dta_trans_heads[l] = 0
            dta_trans_mlps[l] = 0

    results = {
        "static_heads": dta_static_heads.tolist(),
        "static_mlps": dta_static_mlps.tolist(),
        "trans_heads": dta_trans_heads.tolist(),
        "trans_mlps": dta_trans_mlps.tolist(),
    }
    
    os.makedirs("../outputs/generator_analysis", exist_ok=True)
    with open("../outputs/generator_analysis/dta_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("DTA computation complete. Saved to outputs/generator_analysis/dta_results.json")
    
if __name__ == "__main__":
    main()
