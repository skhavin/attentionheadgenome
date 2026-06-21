import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

sys.path.append("phase7")
from audit_heads import load_model, extract_all_V, detect_arch, iter_attn_layers, build_natural_prompts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = load_model(model_name, device)
    
    prompts = build_natural_prompts(tokenizer, 512, 10)
    
    arch = detect_arch(model)
    layers = list(iter_attn_layers(model, arch))
    num_layers = len(layers)
    first_attn = layers[0][1]
    num_heads = getattr(first_attn, "num_heads", getattr(model.config, "num_attention_heads", 1))
    
    # Track average mass per head
    sink_masses = torch.zeros(num_layers, num_heads, device=device)
    local_masses = torch.zeros(num_layers, num_heads, device=device)
    
    num_sink_tokens = 4
    local_window = 64
    
    for ids in prompts:
        V_dict, attn_tuple, logits = extract_all_V(model, ids, arch, device)
        for l_idx in range(num_layers):
            attn_layer = attn_tuple[l_idx].to(device) # [1, H, N, N]
            B, H, N, _ = attn_layer.shape
            
            for h_idx in range(num_heads):
                head_attn = attn_layer[0, h_idx] # [N, N]
                
                # Sink mass
                sm = head_attn[:, :num_sink_tokens].sum(dim=-1).mean()
                sink_masses[l_idx, h_idx] += sm
                
                # Local mass
                lm_sum = 0.0
                for t in range(N):
                    start = max(0, t - local_window + 1)
                    lm_sum += head_attn[t, start:t+1].sum()
                local_masses[l_idx, h_idx] += lm_sum / N
                
    sink_masses /= len(prompts)
    local_masses /= len(prompts)
    
    # Count heads meeting threshold
    sink_heads = []
    local_heads = []
    both_heads = []
    
    for l in range(num_layers):
        for h in range(num_heads):
            sm = sink_masses[l, h].item()
            lm = local_masses[l, h].item()
            
            is_sink = sm > 0.70
            is_local = lm > 0.90
            
            if is_sink:
                sink_heads.append((l, h, sm))
            if is_local:
                local_heads.append((l, h, lm))
            if is_sink and is_local:
                both_heads.append((l, h))
                
    print(f"Total heads: {num_layers * num_heads}")
    print(f"Sink heads (>70% mass on first 4 tokens): {len(sink_heads)} ({len(sink_heads)/(num_layers*num_heads)*100:.1f}%)")
    print(f"Local heads (>90% mass in window 64):     {len(local_heads)} ({len(local_heads)/(num_layers*num_heads)*100:.1f}%)")
    print(f"Both sink and local:                      {len(both_heads)}")
    
    # Print a few examples
    print("\nSample Local heads:")
    for l, h, lm in local_heads[:10]:
        print(f"  Layer {l:2d} Head {h:2d}: Local Mass = {lm:.4f}, Sink Mass = {sink_masses[l,h]:.4f}")
        
    print("\nSample Sink heads:")
    for l, h, sm in sink_heads[:10]:
        print(f"  Layer {l:2d} Head {h:2d}: Sink Mass = {sm:.4f}, Local Mass = {local_masses[l,h]:.4f}")

if __name__ == "__main__":
    main()
