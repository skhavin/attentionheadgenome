import json
import torch
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        val_prompts = json.load(f)
        
    arithmetic_prompts = [p for p in val_prompts if p["task_type"] == "arithmetic"]
    
    # 1. Compute DLA
    print("\n--- 1. Computing Direct Logit Attribution (DLA) ---")
    dla_heads = np.zeros((num_layers, num_heads))
    dla_mlps = np.zeros(num_layers)
    
    prompts = [p["prompt"] for p in arithmetic_prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        target_token_ids = out.sequences[:, -1] # [batch]
        
    target_vectors = model.lm_head.weight[target_token_ids].detach() # [batch, hidden_size]
    
    head_outputs = {l: [] for l in range(num_layers)}
    mlp_outputs = {l: [] for l in range(num_layers)}
    attention_weights = {l: [] for l in range(num_layers)}
    
    def get_o_proj_hook(layer_idx):
        def hook(module, input, output):
            z = input[0][:, -1, :] 
            W_O = module.weight 
            heads_out = []
            for h in range(num_heads):
                z_h = z[:, h*head_dim : (h+1)*head_dim]
                W_O_h = W_O[:, h*head_dim : (h+1)*head_dim]
                head_out = torch.matmul(z_h, W_O_h.T) 
                heads_out.append(head_out.detach().cpu())
            head_outputs[layer_idx].append(torch.stack(heads_out, dim=1))
        return hook
        
    def get_mlp_hook(layer_idx):
        def hook(module, input, output):
            mlp_outputs[layer_idx].append(output[:, -1, :].detach().cpu())
        return hook
        
    def get_attn_hook(layer_idx):
        def hook(module, input, output):
            if len(output) > 1 and output[1] is not None:
                attn = output[1][:, :, -1, :].detach().cpu()
                attention_weights[layer_idx].append(attn)
        return hook

    handles = []
    model.config.output_attentions = True
    
    for l in range(num_layers):
        layer_module = model.model.layers[l]
        handles.append(layer_module.self_attn.o_proj.register_forward_hook(get_o_proj_hook(l)))
        handles.append(layer_module.mlp.register_forward_hook(get_mlp_hook(l)))
        handles.append(layer_module.self_attn.register_forward_hook(get_attn_hook(l)))
        
    with torch.no_grad():
        model(**inputs)
        
    for h in handles:
        h.remove()
    model.config.output_attentions = False
        
    total_dla_mass = 0
    for l in range(num_layers):
        ho = torch.cat(head_outputs[l], dim=0).to(model.device)
        mo = torch.cat(mlp_outputs[l], dim=0).to(model.device)
        
        ho_proj = (ho * target_vectors.unsqueeze(1)).sum(dim=-1).mean(dim=0)
        mo_proj = (mo * target_vectors).sum(dim=-1).mean(dim=0)
        
        dla_heads[l] = ho_proj.float().cpu().numpy()
        dla_mlps[l] = mo_proj.float().cpu().numpy()
        
        total_dla_mass += np.sum(np.abs(dla_heads[l])) + np.abs(dla_mlps[l])
        
    flat_dla = []
    for l in range(num_layers):
        flat_dla.append((f"L{l} MLP", dla_mlps[l], "mlp", l, -1))
        for h in range(num_heads):
            flat_dla.append((f"L{l} H{h}", dla_heads[l, h], "head", l, h))
            
    flat_dla.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n--- Top DLA Components ---")
    top_3_mass = sum(abs(x[1]) for x in flat_dla[:3])
    top_10_mass = sum(abs(x[1]) for x in flat_dla[:10])
    
    print(f"Total Absolute DLA Mass: {total_dla_mass:.2f}")
    print(f"Top-3 Mass: {top_3_mass:.2f} ({top_3_mass/total_dla_mass*100:.1f}%)")
    print(f"Top-10 Mass: {top_10_mass:.2f} ({top_10_mass/total_dla_mass*100:.1f}%)\n")
    
    if top_3_mass / total_dla_mass > 0.5:
        print("H2/H3 CONFIRMED (Top-3 > 50%)")
    elif top_10_mass / total_dla_mass < 0.5:
        print("H4 FAVORED (Top-10 < 50%)")
    else:
        print("AMBIGUOUS RESULT")
        
    for i in range(10):
        print(f"{i+1}. {flat_dla[i][0]:10s} | DLA: {flat_dla[i][1]:.2f}")
        
    print("\n--- 2. Attention Tracing (Top 3 Heads) ---")
    top_heads = [x for x in flat_dla if x[2] == "head"][:3]
    for name, dla, _, l, h in top_heads:
        attn = torch.cat(attention_weights[l], dim=0)[:, h, :]
        p0_len = inputs.attention_mask[0].sum().item()
        p0_attn = attn[0, -p0_len:]
        top_indices = torch.topk(p0_attn, 3).indices
        
        print(f"\n{name} DLA={dla:.2f}")
        print(f"Prompt: {prompts[0]}")
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0, -p0_len:])
        for idx in top_indices:
            token_str = tokens[idx].encode("ascii", "replace").decode("ascii")
            print(f"  Attends to: '{token_str}' (weight: {p0_attn[idx]:.3f})")

    print("\n--- 3. Causal Necessity Test (Mean Ablation) ---")
    
    def evaluate_model(model_eval):
        with torch.no_grad():
            gen = model_eval.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs = tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        target_strings = tokenizer.batch_decode(target_token_ids.unsqueeze(-1), skip_special_tokens=True)
        correct = 0
        for out_str, tgt_str in zip(outputs, target_strings):
            if tgt_str.strip() in out_str.strip():
                correct += 1
                
        acc = correct / len(outputs)
        unique = len(set([s.strip() for s in outputs])) / len(outputs)
        return acc, unique, outputs[:2]

    base_acc, base_uniq, base_outs = evaluate_model(model)
    print(f"Baseline Accuracy: {base_acc*100:.1f}%")
    
    mean_activations = {}
    for name, _, _, l, h in top_heads:
        ho = torch.cat(head_outputs[l], dim=0) 
        mean_act = ho[:, h, :].mean(dim=0) 
        mean_activations[(l, h)] = mean_act.to(model.device)
        
    def get_oproj_ablation_hook(l, h_idx, mean_vec):
        def hook(module, input, output):
            z = input[0]
            W_O = module.weight
            z_h = z[:, :, h_idx*head_dim : (h_idx+1)*head_dim]
            W_O_h = W_O[:, h_idx*head_dim : (h_idx+1)*head_dim]
            head_out = torch.matmul(z_h, W_O_h.T) 
            ablation = mean_vec.unsqueeze(0).unsqueeze(0) - head_out
            return output + ablation
        return hook

    ablation_handles = []
    for name, _, _, l, h in top_heads:
        layer_module = model.model.layers[l]
        ablation_handles.append(layer_module.self_attn.o_proj.register_forward_hook(
            get_oproj_ablation_hook(l, h, mean_activations[(l, h)])
        ))
        
    top_acc, top_uniq, top_outs = evaluate_model(model)
    for h_handle in ablation_handles:
        h_handle.remove()
        
    print(f"\nTop-3 Readout Heads Ablated Accuracy: {top_acc*100:.1f}%")
    print(f"Top-3 Readout Heads Ablated Uniqueness: {top_uniq*100:.1f}%")
    if top_uniq < 0.2:
        print(">>> WARNING: DEGENERATE COLLAPSE DETECTED. Ablation caused stutter-loop, not targeted task failure. <<<")
    print("Sample outputs post-ablation:")
    for o in top_outs: print(f"  '{o}'")
        
    np.random.seed(42)
    random_heads = []
    while len(random_heads) < 3:
        rl = np.random.randint(20, 27)
        rh = np.random.randint(0, num_heads)
        if (rl, rh) not in [(l, h) for _,_,_,l,h in top_heads]:
            random_heads.append((rl, rh))
            
    rand_mean_activations = {}
    for rl, rh in random_heads:
        ho = torch.cat(head_outputs[rl], dim=0) 
        mean_act = ho[:, rh, :].mean(dim=0) 
        rand_mean_activations[(rl, rh)] = mean_act.to(model.device)
        
    rand_ablation_handles = []
    for rl, rh in random_heads:
        layer_module = model.model.layers[rl]
        rand_ablation_handles.append(layer_module.self_attn.o_proj.register_forward_hook(
            get_oproj_ablation_hook(rl, rh, rand_mean_activations[(rl, rh)])
        ))
        
    rand_acc, rand_uniq, rand_outs = evaluate_model(model)
    for h_handle in rand_ablation_handles:
        h_handle.remove()
        
    print(f"\nRandom Control (3 matched heads) Ablated Accuracy: {rand_acc*100:.1f}%")
    print(f"Random Control Uniqueness: {rand_uniq*100:.1f}%")

    print("\n--- 4. MLP Necessity Test on OOD Dataset ---")
    np.random.seed(42)
    ood_prompts = []
    for _ in range(80):
        x = np.random.randint(10, 99)
        y = np.random.randint(10, 99)
        ood_prompts.append(f"What is {x} + {y}?\nAnswer:")
        
    ood_inputs = tokenizer(ood_prompts, return_tensors="pt", padding=True).to(model.device)
    
    # We need to know the correct targets to compute accuracy
    # For a naive check, let's re-run the greedy generation of the unablated model to establish the "ceiling"
    def evaluate_ood(model_eval):
        with torch.no_grad():
            gen = model_eval.generate(**ood_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs = tokenizer.batch_decode(gen[:, ood_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return outputs
        
    base_ood_outs = evaluate_ood(model)
    # The true answer is just x + y
    correct = 0
    np.random.seed(42) # reset seed to re-generate targets
    for o_str in base_ood_outs:
        x = np.random.randint(10, 99)
        y = np.random.randint(10, 99)
        ans = str(x + y)
        if ans in o_str:
            correct += 1
    base_ood_acc = correct / len(base_ood_outs)
    print(f"Baseline OOD Accuracy: {base_ood_acc*100:.1f}%")
    
    # We need to compute the mean activation of MLPs on the OOD set first!
    # Let's hook all MLPs for the OOD set
    ood_mlp_outputs = {l: [] for l in range(num_layers)}
    def get_ood_mlp_hook(layer_idx):
        def hook(module, input, output):
            ood_mlp_outputs[layer_idx].append(output[:, -1, :].detach().cpu())
        return hook
        
    ood_handles = []
    for l in range(num_layers):
        ood_handles.append(model.model.layers[l].mlp.register_forward_hook(get_ood_mlp_hook(l)))
    with torch.no_grad():
        model(**ood_inputs)
    for h in ood_handles: h.remove()
    
    ood_mlp_means = {}
    for l in range(num_layers):
        mo = torch.cat(ood_mlp_outputs[l], dim=0) # [80, hidden]
        ood_mlp_means[l] = mo.mean(dim=0).to(model.device)
        
    def get_mlp_ablation_hook(mean_vec):
        def hook(module, input, output):
            # output is [batch, seq, hidden]
            # mean_vec is [hidden]
            ablation = mean_vec.unsqueeze(0).unsqueeze(0) - output
            return output + ablation
        return hook
        
    # Top 3 MLPs from DLA: L27, L26, L22
    top_mlps = [27, 26, 22]
    
    # Group Ablation (Top 3)
    mlp_group_handles = []
    for l in top_mlps:
        mlp_group_handles.append(model.model.layers[l].mlp.register_forward_hook(get_mlp_ablation_hook(ood_mlp_means[l])))
        
    group_ood_outs = evaluate_ood(model)
    for h in mlp_group_handles: h.remove()
    
    group_correct = 0
    np.random.seed(42)
    for o_str in group_ood_outs:
        x = np.random.randint(10, 99)
        y = np.random.randint(10, 99)
        if str(x + y) in o_str: group_correct += 1
    print(f"Top-3 MLPs Group Ablation OOD Accuracy: {group_correct/len(group_ood_outs)*100:.1f}%")
    
    # Individual Ablation (Top 1: L27)
    ind_handle = model.model.layers[27].mlp.register_forward_hook(get_mlp_ablation_hook(ood_mlp_means[27]))
    ind_ood_outs = evaluate_ood(model)
    ind_handle.remove()
    
    ind_correct = 0
    np.random.seed(42)
    for o_str in ind_ood_outs:
        x = np.random.randint(10, 99)
        y = np.random.randint(10, 99)
        if str(x + y) in o_str: ind_correct += 1
    print(f"Top-1 MLP (L27) Individual Ablation OOD Accuracy: {ind_correct/len(ind_ood_outs)*100:.1f}%")
    
    # Random Control (3 MLPs)
    rand_mlps = [21, 23, 24]
    rand_group_handles = []
    for l in rand_mlps:
        rand_group_handles.append(model.model.layers[l].mlp.register_forward_hook(get_mlp_ablation_hook(ood_mlp_means[l])))
    rand_ood_outs = evaluate_ood(model)
    for h in rand_group_handles: h.remove()
    
    rand_correct = 0
    np.random.seed(42)
    for o_str in rand_ood_outs:
        x = np.random.randint(10, 99)
        y = np.random.randint(10, 99)
        if str(x + y) in o_str: rand_correct += 1
    print(f"Random 3 MLPs Control OOD Accuracy: {rand_correct/len(rand_ood_outs)*100:.1f}%")

if __name__ == "__main__":
    main()
