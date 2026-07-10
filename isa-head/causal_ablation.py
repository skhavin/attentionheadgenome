import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_canonical_labels(model_name):
    label_map = {
        "Qwen/Qwen2.5-1.5B": "Qwen-1.5B",
        "unsloth/Llama-3.2-1B": "Llama-3.2-1B"
    }
    key = label_map.get(model_name, model_name)
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "canonical_labels.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("models", {}).get(key, {}).get("heads", {})

def get_first_token_id(tokenizer, text):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokens[0] if tokens else None

def get_layer_target_heads(layer_idx, canonical_dict, n_heads, target_labels):
    heads_to_ablate = []
    for h in range(n_heads):
        head_key = f"{layer_idx}_{h}"
        label = canonical_dict.get(head_key, {}).get("label", "unknown")
        if label in target_labels:
            heads_to_ablate.append(h)
    return heads_to_ablate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", type=str, default="dataset_60.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    canonical_dict = load_canonical_labels(args.model_name)
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    lm_head = model.lm_head
    final_norm = model.model.norm
    layers = model.model.layers
    
    results = {"fact_recall": [], "niah": []} # We only ablate successful tasks (Pattern Induction failed mostly)
    
    causal_breaks = 0
    total_ablated = 0

    for item in tqdm([d for d in dataset if d["task_type"] in ["fact_recall", "niah"]], desc="Running Ablations"):
        prompt = item["prompt"]
        task_type = item["task_type"]
        target_str = item["target_full"] if task_type == "niah" else item["target"]
        target_token_id = get_first_token_id(tokenizer, target_str)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # 1. Clean Run to find Shift Layer
        hidden_states_per_layer = {}
        hooks = []
        def get_hook(l_idx):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hidden_states_per_layer[l_idx] = hs[0, -1, :].detach().clone()
            return hook
            
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(get_hook(i)))
            
        with torch.no_grad():
            _ = model(**inputs)
            
        for hook in hooks:
            hook.remove()
            
        shift_layer = -1
        for i in range(n_layers):
            hs = hidden_states_per_layer[i]
            logits = lm_head(final_norm(hs))
            pred = torch.argmax(logits, dim=-1).item()
            if pred == target_token_id:
                shift_layer = i
                break
                
        # Only proceed if the model actually successfully shifted
        if shift_layer == -1:
            continue
            
        # 2. Identify heads to ablate in shift_layer and shift_layer-1
        target_labels = ["retrieval", "induction"]
        ablated_heads_dict = {}
        
        heads_shift = get_layer_target_heads(shift_layer, canonical_dict, n_heads, target_labels)
        if heads_shift: ablated_heads_dict[shift_layer] = heads_shift
        
        prev_layer = max(0, shift_layer - 1)
        heads_prev = get_layer_target_heads(prev_layer, canonical_dict, n_heads, target_labels)
        if heads_prev: ablated_heads_dict[prev_layer] = heads_prev
        
        if not ablated_heads_dict:
            continue # No targeted heads to ablate here
            
        total_ablated += 1
            
        # 3. Ablated Run
        ablation_hooks = []
        ablation_hidden_states = {}
        
        def get_ablation_hook(l_idx, heads_to_zero):
            def hook(module, inp, out):
                # out is the output of o_proj. But wait!
                # It's much easier to hook v_proj or the input to o_proj.
                # inp[0] to o_proj is the concatenated V vectors.
                v_concat = inp[0].clone()
                for h in heads_to_zero:
                    v_concat[:, :, h * head_dim : (h+1) * head_dim] = 0.0
                
                # Recompute o_proj manually with zeroed V vectors
                return torch.nn.functional.linear(v_concat, module.weight, module.bias)
            return hook
            
        def get_hs_hook(l_idx):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                ablation_hidden_states[l_idx] = hs[0, -1, :].detach().clone()
            return hook
            
        for l_idx, heads in ablated_heads_dict.items():
            ablation_hooks.append(layers[l_idx].self_attn.o_proj.register_forward_hook(get_ablation_hook(l_idx, heads)))
            
        for i, layer in enumerate(layers):
            ablation_hooks.append(layer.register_forward_hook(get_hs_hook(i)))
            
        with torch.no_grad():
            _ = model(**inputs)
            
        for hook in ablation_hooks:
            hook.remove()
            
        # 4. Check Ablated Results
        ablated_shift_layer = -1
        for i in range(n_layers):
            hs = ablation_hidden_states[i]
            logits = lm_head(final_norm(hs))
            pred = torch.argmax(logits, dim=-1).item()
            if pred == target_token_id:
                ablated_shift_layer = i
                break
                
        final_hs = ablation_hidden_states[n_layers-1]
        final_logits = lm_head(final_norm(final_hs))
        final_pred = torch.argmax(final_logits, dim=-1).item()
        final_success = (final_pred == target_token_id)
        
        causally_broken = False
        if not final_success or (ablated_shift_layer > shift_layer + 1):
            causally_broken = True
            causal_breaks += 1
            
        results[task_type].append({
            "prompt": prompt,
            "original_shift": shift_layer,
            "ablated_shift": ablated_shift_layer,
            "final_success": final_success,
            "ablated_heads": ablated_heads_dict,
            "causally_broken": causally_broken
        })

    print(f"\n--- Causal Ablation Complete ---")
    print(f"Total Prompts Tested (where target heads were present): {total_ablated}")
    print(f"Causal Breaks (Prediction failed or shifted late): {causal_breaks}")
    if total_ablated > 0:
        print(f"Causal Efficacy: {(causal_breaks / total_ablated) * 100:.2f}%")
        
    with open("causal_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
