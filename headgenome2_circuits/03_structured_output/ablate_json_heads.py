import json
import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase2_circuits"
DATASET_PATH = "headgenome2_circuits/datasets/json_brackets.json"

class AblationHook:
    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []
        
    def create_hook(self, target_heads):
        def hook(module, input):
            x = input[0] # input to o_proj, shape: (batch, seq, hidden_dim)
            for h in target_heads:
                start_idx = h * self.head_dim
                end_idx = start_idx + self.head_dim
                x[:, :, start_idx:end_idx] = 0.0
            return (x,)
        return hook

    def register(self, model, target_heads):
        layer_to_heads = {}
        for l, h in target_heads:
            if l not in layer_to_heads:
                layer_to_heads[l] = []
            layer_to_heads[l].append(h)
            
        for layer_idx, heads_in_layer in layer_to_heads.items():
            layer_module = model.model.layers[layer_idx].self_attn.o_proj
            handle = layer_module.register_forward_pre_hook(self.create_hook(heads_in_layer))
            self.hooks.append(handle)
            
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def evaluate_json_validity(model, tokenizer, dataset):
    valid_count = 0
    for item in tqdm(dataset, desc="Evaluating"):
        prompt = item["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=5, # Should be enough for "}}}"
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Get only the generated tokens
        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # We don't care about perfect JSON, just if it attempts to close the nesting
        if '}' in generated:
            valid_count += 1
            
    return valid_count / len(dataset)

def find_matched_null_heads(target_heads, entropy_matrix, num_heads):
    null_heads = []
    # If entropy matrix contains NaNs, replace with 0.0
    import math
    for (L, H) in target_heads:
        target_entropy = entropy_matrix[L][H]
        if math.isnan(target_entropy):
            target_entropy = 0.0
            
        best_null = None
        min_diff = float('inf')
        
        # Search adjacent layers for best match
        search_layers = [L]
        if L > 0: search_layers.append(L-1)
        if L < len(entropy_matrix) - 1: search_layers.append(L+1)
        
        for search_L in search_layers:
            for search_H in range(num_heads):
                if (search_L, search_H) in target_heads or (search_L, search_H) in null_heads:
                    continue
                
                val = entropy_matrix[search_L][search_H]
                if math.isnan(val): val = 0.0
                
                diff = abs(val - target_entropy)
                if diff < min_diff:
                    min_diff = diff
                    best_null = (search_L, search_H)
                    
        if best_null:
            null_heads.append(best_null)
    return null_heads

def run_ablation(model_key="qwen-0.5b", k_heads=4):
    print("Loading model and datasets...")
    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=False, output_hidden_states=False)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    with open(os.path.join(OUTPUT_DIR, f"json_heads_{model_key}.json"), "r") as f:
        probe_data = json.load(f)
        top_heads = probe_data["top_heads"][:k_heads]
        
    with open(os.path.join(OUTPUT_DIR, f"head_entropy_{model_key}.json"), "r") as f:
        entropy_matrix = json.load(f)["entropy_matrix"]
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    target_heads = [(h["layer"], h["head"]) for h in top_heads]
    null_heads = find_matched_null_heads(target_heads, entropy_matrix, num_heads)
    
    print(f"Target JSON Heads: {target_heads}")
    print(f"Matched Null Heads: {null_heads}")
    
    # 1. Baseline
    print("\nEvaluating Baseline...")
    acc_baseline = evaluate_json_validity(model, tokenizer, dataset)
    print(f"Baseline Validity: {acc_baseline*100:.1f}%")
    
    # 2. Null Ablation
    print("\nEvaluating Null Ablation...")
    hook_system = AblationHook(num_heads, head_dim)
    hook_system.register(model, null_heads)
    acc_null = evaluate_json_validity(model, tokenizer, dataset)
    hook_system.remove()
    print(f"Null Ablation Validity: {acc_null*100:.1f}%")
    
    # 3. Target Ablation
    print("\nEvaluating JSON Head Ablation...")
    hook_system.register(model, target_heads)
    acc_target = evaluate_json_validity(model, tokenizer, dataset)
    hook_system.remove()
    print(f"JSON Ablation Validity: {acc_target*100:.1f}%")
    
    # Falsification check
    passed = (acc_null - acc_target) > 0.10
    
    results = {
        "model": model_key,
        "k_heads": k_heads,
        "baseline_acc": acc_baseline,
        "null_acc": acc_null,
        "target_acc": acc_target,
        "target_heads": target_heads,
        "null_heads": null_heads,
        "falsification_passed": passed
    }
    
    with open(os.path.join(OUTPUT_DIR, f"json_ablation_{model_key}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results!")

if __name__ == "__main__":
    run_ablation("qwen-0.5b")
