import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True
    )
    model.eval()

    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Layer 0 attention
    attn_l0 = outputs.attentions[0]  # [B, H, N, N]
    B, H, N, _ = attn_l0.shape
    
    print(f"Layer 0 shape: {attn_l0.shape}")
    
    num_sink_tokens = 4
    local_window = 64
    
    for h in range(H):
        head_attn = attn_l0[0, h]  # [N, N]
        
        # Average sink mass across all queries
        sink_mass = head_attn[:, :num_sink_tokens].sum(dim=-1).mean().item()
        
        # Average local mass across all queries
        # For each query t, we look at [max(0, t - local_window + 1): t + 1]
        local_masses = []
        for t in range(N):
            start = max(0, t - local_window + 1)
            local_masses.append(head_attn[t, start:t+1].sum().item())
        local_mass = sum(local_masses) / len(local_masses)
        
        print(f"Head {h:2d}: Sink Mass = {sink_mass:.4f}, Local Mass = {local_mass:.4f}")

if __name__ == "__main__":
    main()
