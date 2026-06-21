import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

sys.path.append("phase7")
from audit_heads import load_model, extract_all_V
from substitutes import local_substitute

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = load_model(model_name, device)
    
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    V_dict, attn_tuple, logits = extract_all_V(model, ids, "llama", device)
    
    # Layer 0 Head 0
    V_head = V_dict[0][:, :, 0, :].to(device) # [1, N, d_head]
    attn_w = attn_tuple[0][:, 0, :, :].to(device) # [1, N, N]
    
    V4d = V_head.unsqueeze(1)
    attn_out_full = torch.bmm(attn_w, V_head).unsqueeze(1)
    
    print("Layer 0 Head 0 output L2 norm of difference and KL proxy:")
    for W in [2, 4, 8, 16, 32, 64, 128]:
        attn_out_sub = local_substitute(V4d, window_size=W)
        delta_norm = (attn_out_full - attn_out_sub).norm(dim=-1).mean().item()
        kl_approx = delta_norm ** 2 / 2.0
        print(f"  Window W={W:3d}: delta_norm = {delta_norm:.5f}, kl_approx = {kl_approx:.6f}")

if __name__ == "__main__":
    main()
