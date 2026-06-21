import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

sys.path.append("phase7")
from audit_heads import load_model, extract_all_V, relative_linf
from substitutes import local_substitute

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = load_model(model_name, device)
    
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    V_dict, attn_tuple, logits = extract_all_V(model, ids, "llama", device)
    
    # Let's test Layer 0 Head 0 (which we know is highly local)
    V_head = V_dict[0][:, :, 0, :].to(device) # [1, N, d_head]
    attn_w = attn_tuple[0][:, 0, :, :].to(device) # [1, N, N]
    
    V4d = V_head.unsqueeze(1)
    attn_out_full = torch.bmm(attn_w, V_head).unsqueeze(1)
    
    print("Layer 0 Head 0 output relative Linf error:")
    for W in [2, 4, 8, 16, 32, 64, 128]:
        attn_out_sub = local_substitute(V4d, window_size=W)
        err = relative_linf(attn_out_full, attn_out_sub)
        print(f"  Window W={W:3d}: relative Linf = {err:.5f}")

if __name__ == "__main__":
    main()
