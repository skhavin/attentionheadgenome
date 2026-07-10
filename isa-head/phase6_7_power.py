import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cliffs_delta(lst1, lst2):
    m = len(lst1)
    n = len(lst2)
    dom = 0
    for x in lst1:
        for y in lst2:
            if x > y: dom += 1
            elif x < y: dom -= 1
    return dom / (m * n)

def run_edges(dataset_file, edges, model, tokenizer, retrieval_heads):
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
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

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    edge_restorations = {k: [] for k in edges}
    placebo_restorations = {k: [] for k in edges}

    def get_logit_diff(logits, t1, t2):
        return (logits[0, -1, t1] - logits[0, -1, t2]).item()

    for clean_prompt, corrupted_prompt, t_clean, t_corr in tqdm(pairs, desc=f"Evaluating {dataset_file}"):
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
            
        for edge_key in edges:
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

    return edge_restorations, placebo_restorations

def calculate_power(diffs, n_samples=20, n_simulations=5000, alpha=0.05):
    # Diffs is the distribution of (True - Placebo) on the Discovery set.
    mean = np.mean(diffs)
    std = np.std(diffs)
    if std == 0: return 0.0
    sig_count = 0
    for _ in range(n_simulations):
        sample = np.random.normal(loc=mean, scale=std, size=n_samples)
        try:
            _, p_val = stats.wilcoxon(sample, alternative='greater')
            if p_val < alpha:
                sig_count += 1
        except:
            pass
    return sig_count / n_simulations

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Loading model: {model_name} on {DEVICE} for Phase 6/7 Power Analysis")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    with open("phase2_retrieval_heads.json", "r") as f:
        retrieval_heads = json.load(f)

    with open("phase6_7_edges.json", "r") as f:
        top_edges = json.load(f)

    print("\nRunning Discovery Set (N=40) to estimate effect size...")
    disc_true, disc_plac = run_edges("dataset_discovery_40.json", top_edges, model, tokenizer, retrieval_heads)
    
    print("\nRunning Confirmation Set (N=20) for Cliff's Delta...")
    conf_true, conf_plac = run_edges("dataset_confirmation_20.json", top_edges, model, tokenizer, retrieval_heads)
    
    print("\n=== Phase 6/7 Statistical Analysis ===")
    
    all_powers = []
    
    for edge in top_edges:
        print(f"\nEdge {edge}:")
        
        t_conf = conf_true[edge]
        p_conf = conf_plac[edge]
        delta = cliffs_delta(t_conf, p_conf)
        print(f"  Confirmation Set (N=20) Cliff's Delta: {delta:.3f}")
        
        t_disc = disc_true[edge]
        p_disc = disc_plac[edge]
        diffs = [t - p for t, p in zip(t_disc, p_disc)]
        
        power_20 = calculate_power(diffs, n_samples=20)
        power_40 = calculate_power(diffs, n_samples=40)
        power_80 = calculate_power(diffs, n_samples=80)
        all_powers.append(power_20)
        
        print(f"  Discovery Effect Mean Diff: {np.mean(diffs)*100:.2f}%, Std: {np.std(diffs)*100:.2f}%")
        print(f"  Power at N=20: {power_20*100:.1f}%")
        print(f"  Power at N=40: {power_40*100:.1f}%")
        print(f"  Power at N=80: {power_80*100:.1f}%")
        
    print(f"\nAverage Power at N=20 across top 5 edges: {np.mean(all_powers)*100:.1f}%")

if __name__ == "__main__":
    main()
