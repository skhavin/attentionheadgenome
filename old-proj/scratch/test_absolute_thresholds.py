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
            V_dict, attn_tuple, logits = extract_all_V(model, ids, arch, device)
            
            # Precompute query/key indices for vectorized local implied attention
            N = ids.shape[1]
            q_indices = torch.arange(N, device=device).unsqueeze(1) # [N, 1]
            k_indices = torch.arange(N, device=device).unsqueeze(0) # [1, N]
            
            for l_idx in range(num_layers):
                V_layer = V_dict[l_idx].to(device)
                attn_layer = attn_tuple[l_idx].to(device)
                for h_idx in range(num_heads):
                    V_head = V_layer[:, :, h_idx, :]
                    attn_w = attn_layer[:, h_idx, :, :]
                    V4d = V_head.unsqueeze(1)
                    attn_out_full = torch.bmm(attn_w, V_head).unsqueeze(1)
                    
                    for htype in ["sink", "local"]:
                        if htype == "sink":
                            attn_out_sub = sink_substitute(V4d, attn_weights=attn_w.unsqueeze(1), num_sink_tokens=4)
                            # implied attn
                            implied_attn = torch.zeros_like(attn_w)
                            sink_weights = attn_w[:, :, :4].mean(dim=1, keepdim=True)
                            implied_attn[:, :, :4] = sink_weights
                            attn_linf = (attn_w - implied_attn).abs().max().item()
                        else:
                            attn_out_sub = local_substitute(V4d, window_size=64)
                            W = 64
                            mask = (k_indices >= q_indices - W + 1) & (k_indices <= q_indices)
                            local_lens = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                            implied_attn = (mask.float() / local_lens.float()).unsqueeze(0) # [1, N, N]
                            attn_linf = (attn_w - implied_attn).abs().max().item()
                            
                        # ABSOLUTE output L-infinity
                        out_linf = (attn_out_full - attn_out_sub).abs().max().item()
                        
                        delta_norm = (attn_out_full - attn_out_sub).norm(dim=-1).mean().item()
                        kl = delta_norm ** 2 / 2.0
                        
                        metrics = head_metrics[(l_idx, h_idx, htype)]
                        if mode == "nat":
                            metrics["attn_linf_nat"] = max(metrics["attn_linf_nat"], attn_linf)
                            metrics["out_linf_nat"] = max(metrics["out_linf_nat"], out_linf)
                            metrics["kl_nat"] = max(metrics["kl_nat"], kl)
                        else:
                            metrics["attn_linf_copy"] = max(metrics["attn_linf_copy"], attn_linf)
                            metrics["out_linf_copy"] = max(metrics["out_linf_copy"], out_linf)
                            metrics["kl_copy"] = max(metrics["kl_copy"], kl)
                            
    def eval_rule(name, rule_fn):
        t1, t2, t3 = 0, 0, 0
        for (l, h, htype), m in head_metrics.items():
            nat_safe, copy_safe = rule_fn(m, htype)
            if nat_safe and copy_safe:
                t1 += 1
            elif nat_safe and not copy_safe:
                t2 += 1
            else:
                t3 += 1
        total = len(head_metrics)
        print(f"{name:50s} -> Tier 1: {t1:3d} ({t1/total*100:4.1f}%), Tier 2: {t2:3d} ({t2/total*100:4.1f}%), Tier 3: {t3:3d}")

    # Rule 1: User's exact rule
    def rule1(m, htype):
        nat_safe = (m["attn_linf_nat"] < 0.001 and m["out_linf_nat"] < 0.015 and m["kl_nat"] < 0.010)
        copy_safe = (m["attn_linf_copy"] < 0.001 and m["out_linf_copy"] < 0.015 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 1: User's exact (attn < 0.001, out < 0.015)", rule1)

    # Rule 2: Ignore attn_linf on local
    def rule2(m, htype):
        attn_nat_ok = (m["attn_linf_nat"] < 0.001) if htype == "sink" else True
        attn_copy_ok = (m["attn_linf_copy"] < 0.001) if htype == "sink" else True
        nat_safe = (attn_nat_ok and m["out_linf_nat"] < 0.015 and m["kl_nat"] < 0.010)
        copy_safe = (attn_copy_ok and m["out_linf_copy"] < 0.015 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 2: Ignore attn_linf on local (out < 0.015)", rule2)

    # Rule 3: Relax attn_linf to 0.005 on sink, ignore on local
    def rule3(m, htype):
        attn_nat_ok = (m["attn_linf_nat"] < 0.005) if htype == "sink" else True
        attn_copy_ok = (m["attn_linf_copy"] < 0.005) if htype == "sink" else True
        nat_safe = (attn_nat_ok and m["out_linf_nat"] < 0.015 and m["kl_nat"] < 0.010)
        copy_safe = (attn_copy_ok and m["out_linf_copy"] < 0.015 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 3: Relax attn_linf to 0.005 on sink, ignore on local", rule3)

    # Rule 4: Ignore attn_linf completely
    def rule4(m, htype):
        nat_safe = (m["out_linf_nat"] < 0.015 and m["kl_nat"] < 0.010)
        copy_safe = (m["out_linf_copy"] < 0.015 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 4: Ignore attn_linf completely (out < 0.015)", rule4)

    # Rule 5: Ignore attn_linf completely, raise output threshold to 0.05
    def rule5(m, htype):
        nat_safe = (m["out_linf_nat"] < 0.05 and m["kl_nat"] < 0.010)
        copy_safe = (m["out_linf_copy"] < 0.05 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 5: Ignore attn_linf completely, out < 0.05", rule5)

    # Rule 6: Ignore attn_linf completely, raise output threshold to 0.10
    def rule6(m, htype):
        nat_safe = (m["out_linf_nat"] < 0.10 and m["kl_nat"] < 0.010)
        copy_safe = (m["out_linf_copy"] < 0.10 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 6: Ignore attn_linf completely, out < 0.10", rule6)

    # Rule 7: Ignore attn_linf completely, out < 0.15
    def rule7(m, htype):
        nat_safe = (m["out_linf_nat"] < 0.15 and m["kl_nat"] < 0.010)
        copy_safe = (m["out_linf_copy"] < 0.15 and m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 7: Ignore attn_linf completely, out < 0.15", rule7)

    # Rule 8: Ignore output threshold completely (only check KL < 0.01)
    def rule8(m, htype):
        nat_safe = (m["kl_nat"] < 0.010)
        copy_safe = (m["kl_copy"] < 0.010)
        return nat_safe, copy_safe
    eval_rule("Rule 8: Ignore output threshold completely (KL < 0.01)", rule8)

if __name__ == "__main__":
    main()
