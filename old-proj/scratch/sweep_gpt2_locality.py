import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os
from tqdm import tqdm
import math
import functools

print = functools.partial(print, flush=True)

sys.path.append("phase7")
from audit_heads import extract_all_V, detect_arch, iter_attn_layers, build_natural_prompts
from regime_detector import RegimeSwitchingPatcher
from eval_ppl import load_wikitext_test, compute_ppl_chunk

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2-medium"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- STEP 1. Compute Locality ---
    print("\n--- STEP 1: Profiling Locality ---")
    prompts = build_natural_prompts(tokenizer, seq_len=512, num_docs=10)
    arch = detect_arch(model)
    layers = list(iter_attn_layers(model, arch))
    num_layers = len(layers)
    first_attn = layers[0][1]
    num_heads = getattr(first_attn, "num_heads", getattr(model.config, "num_attention_heads", 1))
    
    local_masses = torch.zeros(num_layers, num_heads, device=device)
    local_window = 64
    
    with torch.no_grad():
        for ids in tqdm(prompts, desc="Profiling locality"):
            ret = extract_all_V(model, ids, arch, device)
            attn_tuple = ret[1]
            for l_idx in range(num_layers):
                attn_layer = attn_tuple[l_idx].to(device) # [1, H, N, N]
                B, H, N, _ = attn_layer.shape
                
                q_idx = torch.arange(N, device=device).unsqueeze(1)
                k_idx = torch.arange(N, device=device).unsqueeze(0)
                mask = (k_idx >= q_idx - local_window + 1) & (k_idx <= q_idx)
                
                for h_idx in range(num_heads):
                    head_attn = attn_layer[0, h_idx] # [N, N]
                    mass = (head_attn * mask).sum().item() / N
                    local_masses[l_idx, h_idx] += mass
                    
    local_masses /= len(prompts)
    
    all_heads = []
    for l in range(num_layers):
        for h in range(num_heads):
            all_heads.append({"layer": l, "head": h, "locality": local_masses[l, h].item()})
            
    all_heads = sorted(all_heads, key=lambda x: x["locality"], reverse=True)
    
    print("\nTop 10 heads by locality:")
    for i, h in enumerate(all_heads[:10]):
        print(f"  #{i+1}: L{h['layer']}H{h['head']} locality = {h['locality']:.4f}")
        
    # --- STEP 2. Single-head Sweep ---
    print("\n--- STEP 2: Sweeping top heads for PPL delta ---")
    chunks = load_wikitext_test(tokenizer, seq_len=1024, stride=512, num_docs=40)
    
    def eval_ppl(patcher=None):
        ppls = []
        with torch.no_grad():
            for chunk in chunks:
                ppl = compute_ppl_chunk(model, chunk, device)
                if ppl is not None:
                    ppls.append(ppl)
        return np.mean(ppls)
        
    print("Evaluating baseline (40 docs)...")
    baseline_ppl = eval_ppl(None)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    for h in all_heads[:25]:  # Sweep top 25 heads
        tier1 = [(h["layer"], h["head"], "local")]
        patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1, tier2_heads=[])
        ppl = eval_ppl(patcher)
        delta = ppl - baseline_ppl
        status = "✓" if delta < 0.1 else "✗"
        print(f"L{h['layer']}H{h['head']}: locality={h['locality']:.4f}, ΔPPL={delta:+.3f} {status}")
        patcher.restore()
        
    # --- STEP 3. Multi-Head Substitution ---
    print("\n--- STEP 3: Multi-Head Simultaneous Substitution ---")
    chunks = load_wikitext_test(tokenizer, seq_len=1024, stride=512, num_docs=100)
    
    print("Evaluating baseline (100 docs)...")
    baseline_ppl_100 = eval_ppl(None)
    
    thresholds_to_test = [0.99, 0.98, 0.95, 0.90, 0.85]
    
    for t in thresholds_to_test:
        qualifying = [h for h in all_heads if h["locality"] >= t]
        if len(qualifying) == 0:
            continue
            
        tier1 = [(h["layer"], h["head"], "local") for h in qualifying]
        patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1, tier2_heads=[])
        ppl = eval_ppl(patcher)
        delta = ppl - baseline_ppl_100
        patcher.restore()
        
        status = "✓ SAFE" if delta < 0.5 else "✗ BROKEN"
        print(f"Threshold >= {t:.2f}: {len(qualifying):2d} heads substituted | Cumulative ΔPPL = {delta:+.3f} {status}")

if __name__ == "__main__":
    main()
