import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    with open("dataset_confirmation_20.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    with open("phase2_retrieval_heads.json", "r") as f:
        retrieval_heads = json.load(f)
        
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    head_dla_vals = []
    mlp_dla_vals = []
    
    # We will compare the Top 5 Retrieval Heads vs the late-stage MLPs (e.g. L20 to L27)
    target_mlp_layers = list(range(20, n_layers))

    for item in tqdm(prompts, desc="Phase 4 (MLP vs Attention DLA)"):
        tokens = tokenizer(item["prompt"], return_tensors="pt").to(DEVICE)
        target_token = item.get("target_full", item.get("target"))
        target_id = tokenizer(target_token, add_special_tokens=False).input_ids[0]
        
        head_cache = {}
        def head_hook_fn(module, args, layer_idx):
            head_cache[layer_idx] = args[0][0, -1, :].detach().clone()
            
        mlp_cache = {}
        def mlp_hook_fn(module, args, output, layer_idx):
            mlp_cache[layer_idx] = output[0, -1, :].detach().clone()
        
        handles = []
        for l in range(n_layers):
            h_attn = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, l_idx=l: head_hook_fn(m, a, l_idx)
            )
            h_mlp = model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o, l_idx=l: mlp_hook_fn(m, a, o, l_idx)
            )
            handles.extend([h_attn, h_mlp])
            
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            
        for h in handles: h.remove()
        
        # 1. Attention Head DLA
        prompt_head_dlas = []
        for rh in retrieval_heads:
            l, h_idx = rh["layer"], rh["head"]
            head_vec = head_cache[l][h_idx * d_head : (h_idx + 1) * d_head]
            full_vec = torch.zeros(d_model, dtype=model.dtype, device=DEVICE)
            full_vec[h_idx * d_head : (h_idx + 1) * d_head] = head_vec
            
            with torch.no_grad():
                head_out = model.model.layers[l].self_attn.o_proj(full_vec)
                dla_logits = model.lm_head(model.model.norm(head_out))
            prompt_head_dlas.append(dla_logits[target_id].item())
            
        # 2. MLP DLA
        prompt_mlp_dlas = []
        for l in target_mlp_layers:
            mlp_out = mlp_cache[l]
            with torch.no_grad():
                dla_logits = model.lm_head(model.model.norm(mlp_out))
            prompt_mlp_dlas.append(dla_logits[target_id].item())
            
        # We take the mean DLA per prompt for both groups
        head_dla_vals.append(np.mean(prompt_head_dlas))
        mlp_dla_vals.append(np.mean(prompt_mlp_dlas))

    mean_head_dla = np.mean(head_dla_vals)
    mean_mlp_dla = np.mean(mlp_dla_vals)
    
    print("\n--- Phase 4 DLA Comparison (N=20) ---")
    print(f"Mean DLA (Retrieval Heads): {mean_head_dla:.4f}")
    print(f"Mean DLA (Late MLPs L20-L27): {mean_mlp_dla:.4f}")
    
    try:
        w_stat, p_val = stats.wilcoxon(mlp_dla_vals, head_dla_vals, alternative='greater')
    except:
        p_val = 1.0
        
    print(f"Wilcoxon p-value (MLP > Head): {p_val:.4e}")

if __name__ == "__main__":
    main()
