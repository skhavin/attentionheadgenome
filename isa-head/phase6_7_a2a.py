import argparse
import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--mode", type=str, choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} on {DEVICE} for Phase 6/7 ({args.mode})")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    dataset_file = "dataset_discovery_40.json" if args.mode == "discovery" else "dataset_confirmation_20.json"
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # Build pairs across all tasks
    task_groups = {}
    for item in dataset:
        tt = item["task_type"]
        if tt not in task_groups: task_groups[tt] = []
        task_groups[tt].append(item)
        
    pairs = []
    for tt, items in task_groups.items():
        if len(items) < 2: continue
        for i in range(len(items)):
            clean = items[i]
            corrupted = items[(i + 1) % len(items)]
            tgt_clean = clean.get("target_full", clean.get("target"))
            tgt_corr = corrupted.get("target_full", corrupted.get("target"))
            t_clean = tokenizer(tgt_clean, add_special_tokens=False).input_ids[0]
            t_corr = tokenizer(tgt_corr, add_special_tokens=False).input_ids[0]
            pairs.append((clean["prompt"], corrupted["prompt"], t_clean, t_corr))
            
    print(f"Loaded {len(pairs)} pairs for Phase 6/7.")

    with open("phase2_retrieval_heads.json", "r") as f:
        retrieval_heads = json.load(f)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    def get_logit_diff(logits, t1, t2):
        return (logits[0, -1, t1] - logits[0, -1, t2]).item()

    if args.mode == "discovery":
        edge_restorations = {}
        
        # Test all heads in L0 to L21 as candidate 'Head A'
        # (Assuming Retrieval Heads are L26-L27)
        cand_layers = list(range(0, 22))
        
        for clean_prompt, corrupted_prompt, t_clean, t_corr in tqdm(pairs, desc="Phase 6/7 (Discovery)"):
            tok_clean = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
            tok_corr = tokenizer(corrupted_prompt, return_tensors="pt").to(DEVICE)
            
            clean_head_resid = {}
            corr_head_resid = {}
            
            # Helper to extract head output into residual stream
            def save_head_resid(layer_idx, x, cache_dict):
                for h_idx in range(n_heads):
                    full_vec = torch.zeros_like(x)
                    full_vec[:, :, h_idx*d_head : (h_idx+1)*d_head] = x[:, :, h_idx*d_head : (h_idx+1)*d_head]
                    # Pass through W_O using F.linear to avoid triggering the hook again
                    w_o = model.model.layers[layer_idx].self_attn.o_proj.weight
                    resid = F.linear(full_vec, w_o)
                    cache_dict[(layer_idx, h_idx)] = resid[0, -1, :].detach().clone()
                    
            clean_handles = []
            for l in cand_layers:
                h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, l_idx=l: save_head_resid(l_idx, a[0], clean_head_resid)
                )
                clean_handles.append(h)
                
            with torch.no_grad():
                clean_logits = model(**tok_clean).logits
            for h in clean_handles: h.remove()
            clean_diff = get_logit_diff(clean_logits, t_clean, t_corr)
            
            corr_handles = []
            for l in cand_layers:
                h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, l_idx=l: save_head_resid(l_idx, a[0], corr_head_resid)
                )
                corr_handles.append(h)
                
            with torch.no_grad():
                corr_logits = model(**tok_corr).logits
            for h in corr_handles: h.remove()
            corr_diff = get_logit_diff(corr_logits, t_clean, t_corr)
            
            diff_denom = clean_diff - corr_diff
            if diff_denom < 1e-4: continue
            
            # Now patch edges from (L_A, H_A) -> Query of All Retrieval Heads
            for l_A in cand_layers:
                for h_A in range(n_heads):
                    edge_key = f"{l_A}_{h_A}"
                    if edge_key not in edge_restorations: edge_restorations[edge_key] = []
                    
                    delta_resid = clean_head_resid[(l_A, h_A)] - corr_head_resid[(l_A, h_A)] # (d_model)
                    
                    patch_handles = []
                    
                    # Group retrieval heads by layer to apply one hook per layer
                    rh_by_layer = {}
                    for rh in retrieval_heads:
                        if rh["layer"] not in rh_by_layer: rh_by_layer[rh["layer"]] = []
                        rh_by_layer[rh["layer"]].append(rh["head"])
                        
                    for l_B, h_Bs in rh_by_layer.items():
                        def q_patch_hook(module, args, output, h_Bs=h_Bs):
                            # output is (batch, seq, n_heads * d_head)
                            # module is q_proj
                            w_q = module.weight # (n_heads * d_head, d_model)
                            for h_B in h_Bs:
                                w_q_B = w_q[h_B*d_head : (h_B+1)*d_head, :]
                                delta_q = torch.matmul(delta_resid, w_q_B.T)
                                output[0, -1, h_B*d_head : (h_B+1)*d_head] += delta_q
                            return output
                            
                        h = model.model.layers[l_B].self_attn.q_proj.register_forward_hook(q_patch_hook)
                        patch_handles.append(h)
                        
                    with torch.no_grad():
                        patched_logits = model(**tok_corr).logits
                    for h in patch_handles: h.remove()
                    
                    patched_diff = get_logit_diff(patched_logits, t_clean, t_corr)
                    edge_restorations[edge_key].append((patched_diff - corr_diff) / diff_denom)
                    
        # Sort and save top edges
        results = []
        for edge_key, rests in edge_restorations.items():
            if len(rests) > 0:
                results.append((np.mean(rests), edge_key))
        results.sort(reverse=True)
        
        print("\n--- Phase 6/7 Discovery Complete ---")
        print("Top 5 A2A Edges (Targeting Retrieval Heads):")
        top_edges = []
        for r, key in results[:5]:
            print(f"Head {key}: {r*100:.2f}% restoration")
            top_edges.append(key)
            
        with open("phase6_7_edges.json", "w") as f:
            json.dump(top_edges, f, indent=2)

    else:
        # Confirmation mode
        with open("phase6_7_edges.json", "r") as f:
            top_edges = json.load(f)
            
        edge_restorations = {k: [] for k in top_edges}
        placebo_restorations = {k: [] for k in top_edges}
        
        for clean_prompt, corrupted_prompt, t_clean, t_corr in tqdm(pairs, desc="Phase 6/7 (Confirmation)"):
            tok_clean = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
            tok_corr = tokenizer(corrupted_prompt, return_tensors="pt").to(DEVICE)
            
            clean_head_resid = {}
            corr_head_resid = {}
            
            def save_head_resid(layer_idx, x, cache_dict):
                for h_idx in range(n_heads):
                    full_vec = torch.zeros_like(x)
                    full_vec[:, :, h_idx*d_head : (h_idx+1)*d_head] = x[:, :, h_idx*d_head : (h_idx+1)*d_head]
                    w_o = model.model.layers[layer_idx].self_attn.o_proj.weight
                    resid = F.linear(full_vec, w_o)
                    cache_dict[(layer_idx, h_idx)] = resid[0, -1, :].detach().clone()
                    
            handles = []
            for l in range(22):
                handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, l_idx=l: save_head_resid(l_idx, a[0], clean_head_resid)))
            with torch.no_grad(): clean_logits = model(**tok_clean).logits
            for h in handles: h.remove()
            clean_diff = get_logit_diff(clean_logits, t_clean, t_corr)
            
            handles = []
            for l in range(22):
                handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, l_idx=l: save_head_resid(l_idx, a[0], corr_head_resid)))
            with torch.no_grad(): corr_logits = model(**tok_corr).logits
            for h in handles: h.remove()
            corr_diff = get_logit_diff(corr_logits, t_clean, t_corr)
            
            diff_denom = clean_diff - corr_diff
            if diff_denom < 1e-4: continue
            
            rh_by_layer = {}
            for rh in retrieval_heads:
                if rh["layer"] not in rh_by_layer: rh_by_layer[rh["layer"]] = []
                rh_by_layer[rh["layer"]].append(rh["head"])
                
            for edge_key in top_edges:
                l_A, h_A = map(int, edge_key.split("_"))
                
                # True Patch
                delta_resid = clean_head_resid[(l_A, h_A)] - corr_head_resid[(l_A, h_A)]
                patch_handles = []
                for l_B, h_Bs in rh_by_layer.items():
                    def q_patch_hook(module, args, output, h_Bs=h_Bs, delta_resid=delta_resid):
                        w_q = module.weight
                        for h_B in h_Bs:
                            w_q_B = w_q[h_B*d_head : (h_B+1)*d_head, :]
                            output[0, -1, h_B*d_head : (h_B+1)*d_head] += torch.matmul(delta_resid, w_q_B.T)
                        return output
                    patch_handles.append(model.model.layers[l_B].self_attn.q_proj.register_forward_hook(q_patch_hook))
                with torch.no_grad(): patched_logits = model(**tok_corr).logits
                for h in patch_handles: h.remove()
                edge_restorations[edge_key].append((get_logit_diff(patched_logits, t_clean, t_corr) - corr_diff) / diff_denom)
                
                # Placebo Patch (Random head from a different layer)
                placebo_l = l_A - 5 if l_A >= 5 else l_A + 5
                placebo_h = random.randint(0, n_heads-1)
                delta_resid_placebo = clean_head_resid[(placebo_l, placebo_h)] - corr_head_resid[(placebo_l, placebo_h)]
                patch_handles = []
                for l_B, h_Bs in rh_by_layer.items():
                    def q_patch_hook_placebo(module, args, output, h_Bs=h_Bs, delta_resid=delta_resid_placebo):
                        w_q = module.weight
                        for h_B in h_Bs:
                            w_q_B = w_q[h_B*d_head : (h_B+1)*d_head, :]
                            output[0, -1, h_B*d_head : (h_B+1)*d_head] += torch.matmul(delta_resid, w_q_B.T)
                        return output
                    patch_handles.append(model.model.layers[l_B].self_attn.q_proj.register_forward_hook(q_patch_hook_placebo))
                with torch.no_grad(): placebo_logits = model(**tok_corr).logits
                for h in patch_handles: h.remove()
                placebo_restorations[edge_key].append((get_logit_diff(placebo_logits, t_clean, t_corr) - corr_diff) / diff_denom)

        print("\n--- Phase 6/7 Confirmation Complete ---")
        for key in top_edges:
            trues = edge_restorations[key]
            placebos = placebo_restorations[key]
            mean_true = np.mean(trues)
            mean_placebo = np.mean(placebos)
            try:
                w_stat, p_val = stats.wilcoxon(trues, placebos, alternative='greater')
            except:
                p_val = 1.0
            print(f"Edge {key} -> Retrieval Heads:")
            print(f"  True Restoration: {mean_true*100:.1f}%")
            print(f"  Placebo Restoration: {mean_placebo*100:.1f}%")
            print(f"  p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()
