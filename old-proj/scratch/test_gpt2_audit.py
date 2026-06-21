import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

sys.path.append("phase7")
from audit_heads import detect_arch, iter_attn_layers, build_natural_prompts, build_copy_trigger_prompts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2-medium"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    config._attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    model.eval()
    
    natural_prompts = build_natural_prompts(tokenizer, 512, 5)
    copy_prompts = build_copy_trigger_prompts(tokenizer, 512, 5)
    
    arch = detect_arch(model)
    layers = list(iter_attn_layers(model, arch))
    num_layers = len(layers)
    first_attn = layers[0][1]
    num_heads = getattr(model.config, "n_head", 12)
    
    print(f"Auditing GPT-2 with original audit logic...")
    
    # Store max metrics per head
    head_metrics = {}
    for l in range(num_layers):
        for h in range(num_heads):
            for htype in ["sink", "local"]:
                head_metrics[(l, h, htype)] = {
                    "l_inf_nat": 0.0, "l_inf_copy": 0.0,
                    "kl_nat": 0.0, "kl_copy": 0.0
                }
                
    from phase7.substitutes import sink_substitute, local_substitute
    
    # We need to capture V by registering a hook
    def extract_V_gpt2(model, ids, layer_idx, head_idx):
        V_captured = {}
        def _hook(module, inp, output):
            B, N, d_out = output.shape
            num_heads = getattr(model.config, "n_head", 12)
            d_head = d_out // (3 * num_heads)
            _, _, v = output.split(d_out // 3, dim=2)
            v_heads = v.view(B, N, num_heads, d_head)
            V_captured["V"] = v_heads[:, :, head_idx].detach()
        
        _, attn_module = layers[layer_idx]
        handle = attn_module.c_attn.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(ids)
        finally:
            handle.remove()
        return V_captured.get("V")

    datasets = [("nat", natural_prompts), ("copy", copy_prompts)]
    for mode, prompts in datasets:
        for ids in prompts:
            ids = ids.to(device)
            # Just do first 3 layers, 3 heads to be super fast
            for l_idx in range(min(num_layers, 3)):
                # We need to get attn_w
                with torch.no_grad():
                    out = model(ids, output_attentions=True)
                attn_tuple = out.attentions
                
                for h_idx in range(min(num_heads, 3)):
                    V_head = extract_V_gpt2(model, ids, l_idx, h_idx).to(device)
                    attn_w = attn_tuple[l_idx][:, h_idx].to(device)
                    V4d = V_head.unsqueeze(1)
                    attn_out_full = torch.bmm(attn_w, V_head).unsqueeze(1)
                    
                    for htype in ["sink", "local"]:
                        if htype == "sink":
                            attn_out_sub = sink_substitute(V4d, attn_weights=attn_w.unsqueeze(1), num_sink_tokens=4)
                        else:
                            attn_out_sub = local_substitute(V4d, window_size=64)
                            
                        # Original absolute output L-infinity
                        l_inf = (attn_out_full - attn_out_sub).abs().max().item()
                        
                        delta_norm = (attn_out_full - attn_out_sub).norm(dim=-1).mean().item()
                        kl = delta_norm ** 2 / 2.0
                        
                        metrics = head_metrics[(l_idx, h_idx, htype)]
                        if mode == "nat":
                            metrics["l_inf_nat"] = max(metrics["l_inf_nat"], l_inf)
                            metrics["kl_nat"] = max(metrics["kl_nat"], kl)
                        else:
                            metrics["l_inf_copy"] = max(metrics["l_inf_copy"], l_inf)
                            metrics["kl_copy"] = max(metrics["kl_copy"], kl)
                            
    print("\nGPT-2 audit sample results:")
    for (l, h, htype), m in head_metrics.items():
        if l < 3 and h < 3:
            print(f"L{l}H{h} {htype:5s}: kl_nat={m['kl_nat']:.6f}, kl_copy={m['kl_copy']:.6f}, l_inf_nat={m['l_inf_nat']:.6f}, l_inf_copy={m['l_inf_copy']:.6f}")

if __name__ == "__main__":
    main()
