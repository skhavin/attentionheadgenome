import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

sys.path.append("phase7")
from audit_heads import load_model, extract_all_V, detect_arch, iter_attn_layers
from substitutes import local_substitute

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = load_model(model_name, device)
    
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    arch = detect_arch(model)
    layers = list(iter_attn_layers(model, arch))
    
    V_dict, attn_tuple, logits = extract_all_V(model, ids, arch, device)
    
    with torch.no_grad():
        out_f = model(ids, output_hidden_states=True)
        
    for l_idx in [0, 5, 12, 18, 22]:
        h_idx = 0
        W = 64
        
        V_head = V_dict[l_idx][:, :, h_idx, :].to(device) # [1, N, d_head]
        attn_w = attn_tuple[l_idx][:, h_idx, :, :].to(device) # [1, N, N]
        
        V4d = V_head.unsqueeze(1)
        attn_out_full = torch.bmm(attn_w, V_head) # [1, N, d_head]
        attn_out_sub = local_substitute(V4d, window_size=W).squeeze(1) # [1, N, d_head]
        
        diff_unproj = attn_out_full - attn_out_sub
        delta_unproj = diff_unproj.norm(dim=-1).mean().item()
        kl_old = delta_unproj ** 2 / 2.0
        
        _, attn_module = layers[l_idx]
        o_proj = attn_module.o_proj
        d_head = V_head.shape[-1]
        
        w_slice = o_proj.weight[:, h_idx * d_head : (h_idx + 1) * d_head] # [d_model, d_head]
        diff_proj = torch.matmul(diff_unproj, w_slice.t()) # [1, N, d_model]
        
        delta_proj = diff_proj.norm(dim=-1).mean().item()
        kl_projected = delta_proj ** 2 / 2.0
        
        # Use hidden state norm of the layer output
        hs_norm = out_f.hidden_states[l_idx].norm(dim=-1).mean().item()
        d_model = model.config.hidden_size
        rms_hs = hs_norm / (d_model ** 0.5)
        
        delta_proj_scaled = diff_proj.norm(dim=-1).mean().item() / rms_hs
        kl_scaled = delta_proj_scaled ** 2 / 2.0
        
        print(f"Layer {l_idx:2d} Head {h_idx} Local (W={W}):")
        print(f"  Old KL proxy:       {kl_old:10.6f}")
        print(f"  Projected KL proxy: {kl_projected:10.6f}")
        print(f"  Scaled KL proxy:    {kl_scaled:10.6f}  (hs_norm={hs_norm:.1f}, rms_hs={rms_hs:.3f})")

if __name__ == "__main__":
    main()
