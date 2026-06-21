import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

sys.path.append("phase7")
from audit_heads import load_model, extract_all_V, detect_arch, iter_attn_layers, build_natural_prompts, build_copy_trigger_prompts
from substitutes import local_substitute, sink_substitute

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = load_model(model_name, device)
    
    natural_prompts = build_natural_prompts(tokenizer, 512, 5)
    copy_prompts = build_copy_trigger_prompts(tokenizer, 512, 5)
    
    arch = detect_arch(model)
    layers = list(iter_attn_layers(model, arch))
    num_layers = len(layers)
    first_attn = layers[0][1]
    num_heads = getattr(first_attn, "num_heads", getattr(model.config, "num_attention_heads", 1))
    
    # Store max metrics per head
    head_metrics = {}
    for l in range(num_layers):
        for h in range(num_heads):
            for htype in ["sink", "local"]:
                head_metrics[(l, h, htype)] = {
                    "attn_linf_nat": 0.0, "attn_linf_copy": 0.0,
                    "out_linf_nat": 0.0, "out_linf_copy": 0.0,
                    "kl_nat": 0.0, "kl_copy": 0.0
                }
                
    datasets = [("nat", natural_prompts), ("copy", copy_prompts)]
    for mode, prompts in datasets:
        for ids in prompts:
            ids = ids.to(device)
            V_dict, attn_tuple, logits = extract_all_V(model, ids, arch, device)
            
            # Precompute query/key indices for vectorized local implied attention
            N = ids.shape[1]
            q_indices = torch.arange(N, device=device).unsqueeze(1) # [N, 1]
            k_indices = torch.arange(N, device=device).unsqueeze(0) # [1, N]
            
            # Run model forward pass to capture hidden states (for RMSNorm scaling)
            with torch.no_grad():
                out_f = model(ids, output_hidden_states=True)
                
            for l_idx in range(num_layers):
                V_layer = V_dict[l_idx].to(device)
                attn_layer = attn_tuple[l_idx].to(device)
                
                # Hidden state input to next block is hidden_states[l_idx+1]
                # Let's get its RMS
                hs_layer = out_f.hidden_states[l_idx].to(device) # [1, N, d_model]
                d_model = model.config.hidden_size
                rms_hs = hs_layer.norm(dim=-1).mean().item() / (d_model ** 0.5)
                
                # o_proj weight slice for this layer
                _, attn_module = layers[l_idx]
                o_proj = attn_module.o_proj
                
                for h_idx in range(num_heads):
                    V_head = V_layer[:, :, h_idx, :]
                    attn_w = attn_layer[:, h_idx, :, :]
                    V4d = V_head.unsqueeze(1)
                    attn_out_full = torch.bmm(attn_w, V_head) # [1, N, d_head]
                    
                    for htype in ["sink", "local"]:
                        if htype == "sink":
                            attn_out_sub = sink_substitute(V4d, attn_weights=attn_w.unsqueeze(1), num_sink_tokens=4).squeeze(1)
                            # implied attn
                            implied_attn = torch.zeros_like(attn_w)
                            sink_weights = attn_w[:, :, :4].mean(dim=1, keepdim=True)
                            implied_attn[:, :, :4] = sink_weights
                            attn_linf = (attn_w - implied_attn).abs().max().item()
                        else:
                            attn_out_sub = local_substitute(V4d, window_size=64).squeeze(1)
                            W = 64
                            mask = (k_indices >= q_indices - W + 1) & (k_indices <= q_indices)
                            local_lens = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                            implied_attn = (mask.float() / local_lens.float()).unsqueeze(0) # [1, N, N]
                            attn_linf = (attn_w - implied_attn).abs().max().item()
                            
                        # ABSOLUTE output L-infinity (before o_proj)
                        out_linf = (attn_out_full - attn_out_sub).abs().max().item()
                        
                        # PROJECTED and SCALED KL proxy
                        diff_unproj = attn_out_full - attn_out_sub # [1, N, d_head]
                        d_head = V_head.shape[-1]
                        w_slice = o_proj.weight[:, h_idx * d_head : (h_idx + 1) * d_head] # [d_model, d_head]
                        diff_proj = torch.matmul(diff_unproj, w_slice.t()) # [1, N, d_model]
                        
                        delta_proj_scaled = diff_proj.norm(dim=-1).mean().item() / rms_hs
                        kl = delta_proj_scaled ** 2 / 2.0
                        
                        metrics = head_metrics[(l_idx, h_idx, htype)]
                        if mode == "nat":
                            metrics["attn_linf_nat"] = max(metrics["attn_linf_nat"], attn_linf)
                            metrics["out_linf_nat"] = max(metrics["out_linf_nat"], out_linf)
                            metrics["kl_nat"] = max(metrics["kl_nat"], kl)
                        else:
                            metrics["attn_linf_copy"] = max(metrics["attn_linf_copy"], attn_linf)
                            metrics["out_linf_copy"] = max(metrics["out_linf_copy"], out_linf)
                            metrics["kl_copy"] = max(metrics["kl_copy"], kl)
                            
    # Analyze percentiles of kl_nat and out_linf_nat
    kl_nats = [m["kl_nat"] for m in head_metrics.values()]
    out_linfs = [m["out_linf_nat"] for m in head_metrics.values()]
    
    print("\nScaled KL Natural Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
         print(f"  P{p}: {np.percentile(kl_nats, p):.6f}")
         
    print("\nAbsolute Output Linf Natural Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
         print(f"  P{p}: {np.percentile(out_linfs, p):.6f}")

if __name__ == "__main__":
    main()
