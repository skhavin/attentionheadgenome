import argparse
import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--mode", type=str, choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} on {DEVICE} for Phase 4 ({args.mode})")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    dataset_file = "dataset_discovery_40.json" if args.mode == "discovery" else "dataset_confirmation_20.json"
    with open(dataset_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # In Phase 2, we identified the Top 5 Retrieval Heads
    with open("phase2_retrieval_heads.json", "r") as f:
        retrieval_heads = json.load(f)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    # We need to compute DLA of specific heads.
    # DLA = (head_out @ W_O) @ layer_norm_weights @ lm_head
    
    retrieval_dla = []
    random_dla = []

    for item in tqdm(prompts, desc=f"Phase 4 ({args.mode})"):
        tokens = tokenizer(item["prompt"], return_tensors="pt").to(DEVICE)
        target_token = item.get("target_full", item.get("target"))
        target_id = tokenizer(target_token, add_special_tokens=False).input_ids[0]
        
        # We need to hook the attention output for specific heads
        # Qwen2 attention output is computed as (attn_weights @ V) @ W_O.
        # To get a single head's output, we can hook the `o_proj` input.
        
        cache = {}
        def hook_fn(module, args, layer_idx):
            # args[0] is the output of the attention mixing (batch, seq, hidden_size)
            cache[layer_idx] = args[0][0, -1, :].detach().clone()
        
        handles = []
        for l in range(n_layers):
            h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, l_idx=l: hook_fn(m, a, l_idx)
            )
            handles.append(h)
            
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            
        for h in handles: h.remove()
        
        final_hidden = outputs.hidden_states[-1][0, -1, :]
        
        # We need to approximate the LayerNorm scaling factor at the final layer
        # final_hidden = RMSNorm(resid_post)
        # So we can just run the head vector through the final norm and lm_head!
        
        for rh in retrieval_heads:
            l, h_idx = rh["layer"], rh["head"]
            # Extract the specific head's vector from the concatenated V output
            head_vec = cache[l][h_idx * d_head : (h_idx + 1) * d_head]
            # Zero out other heads to push it through W_O
            full_vec = torch.zeros(d_model, dtype=model.dtype, device=DEVICE)
            full_vec[h_idx * d_head : (h_idx + 1) * d_head] = head_vec
            
            with torch.no_grad():
                head_out = model.model.layers[l].self_attn.o_proj(full_vec)
                # DLA is projection to vocab
                dla_logits = model.lm_head(model.model.norm(head_out))
                
            retrieval_dla.append(dla_logits[target_id].item())
            
            # Select a random head from the same layer as a control
            rand_h_idx = random.choice([x for x in range(n_heads) if x != h_idx])
            rand_head_vec = cache[l][rand_h_idx * d_head : (rand_h_idx + 1) * d_head]
            
            full_rand_vec = torch.zeros(d_model, dtype=model.dtype, device=DEVICE)
            full_rand_vec[rand_h_idx * d_head : (rand_h_idx + 1) * d_head] = rand_head_vec
            
            with torch.no_grad():
                rand_head_out = model.model.layers[l].self_attn.o_proj(full_rand_vec)
                rand_dla_logits = model.lm_head(model.model.norm(rand_head_out))
                
            random_dla.append(rand_dla_logits[target_id].item())

    mean_retrieval = np.mean(retrieval_dla)
    mean_random = np.mean(random_dla)
    
    if args.mode == "discovery":
        print("\n--- Phase 4 Discovery Complete ---")
        print(f"Mean DLA (Retrieval Heads): {mean_retrieval:.4f}")
        print(f"Mean DLA (Random Heads): {mean_random:.4f}")
    else:
        print("\n--- Phase 4 Confirmation Complete ---")
        print(f"Mean DLA (Retrieval Heads): {mean_retrieval:.4f}")
        print(f"Mean DLA (Random Heads): {mean_random:.4f}")
        
        try:
            w_stat, p_val = stats.wilcoxon(retrieval_dla, random_dla, alternative='greater')
        except:
            p_val = 1.0
            
        print(f"Wilcoxon p-value: {p_val:.4e}")
        
        with open("phase4_results.json", "w") as f:
            json.dump({
                "mean_retrieval": float(mean_retrieval),
                "mean_random": float(mean_random),
                "p_val": p_val
            }, f, indent=2)

if __name__ == "__main__":
    main()
