import json
import torch
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase2_circuits"
DATASET_PATH = "headgenome2_circuits/datasets/copy_uuids.json"

class AblationHook:
    def __init__(self, heads_to_ablate, num_heads, head_dim):
        self.heads_to_ablate = heads_to_ablate # list of (layer, head)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []
        
    def create_hook(self, target_heads):
        def hook(module, input):
            # Qwen2 attention output is just a concatenated tensor of all head outputs.
            # Shape: (batch, seq, num_heads * head_dim)
            x = input[0] # input to o_proj
            for h in target_heads:
                start_idx = h * self.head_dim
                end_idx = start_idx + self.head_dim
                x[:, :, start_idx:end_idx] = 0.0
            return (x,) # return modified input to the module
        return hook

    def register(self, model):
        # group heads by layer
        layer_to_heads = {}
        for l, h in self.heads_to_ablate:
            if l not in layer_to_heads:
                layer_to_heads[l] = []
            layer_to_heads[l].append(h)
            
        for layer_idx, target_heads in layer_to_heads.items():
            layer_module = model.model.layers[layer_idx].self_attn.o_proj
            handle = layer_module.register_forward_pre_hook(self.create_hook(target_heads))
            self.hooks.append(handle)
            
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def find_matched_null_heads(copy_heads, entropy_matrix, num_heads_total):
    """Finds null heads matching layer depth and baseline entropy."""
    null_heads = []
    # copy_heads is list of dict: {"layer": L, "head": H, "mass": M}
    # Avoid picking copy heads as null heads!
    copy_set = set((h["layer"], h["head"]) for h in copy_heads)
    
    for ch in copy_heads:
        L = ch["layer"]
        target_entropy = entropy_matrix[L][ch["head"]]
        
        # Search in the same layer or adjacent layers (L-1, L, L+1)
        best_null = None
        min_diff = float("inf")
        
        for search_L in [L, L-1, L+1]:
            if search_L < 0 or search_L >= len(entropy_matrix):
                continue
            for H in range(num_heads_total):
                if (search_L, H) in copy_set or (search_L, H) in null_heads:
                    continue
                
                diff = abs(entropy_matrix[search_L][H] - target_entropy)
                if diff < min_diff:
                    min_diff = diff
                    best_null = (search_L, H)
                    
        if best_null:
            null_heads.append(best_null)
            
    return null_heads

def evaluate_accuracy(model, tokenizer, dataset, max_new_tokens=40):
    correct = 0
    total = len(dataset)
    for item in tqdm(dataset, desc="Evaluating"):
        prompt = item["prompt"]
        target = item["target"].strip()
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        # Check if the target UUID is in the generated text (allow some whitespace padding)
        if target in generated_text:
            correct += 1
            
    return correct / total

def run_ablation(model_key="qwen-0.5b", k_heads=4):
    print("Loading model and datasets...")
    model, tokenizer = load_model_and_tokenizer(model_key)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    with open(os.path.join(OUTPUT_DIR, f"copy_heads_{model_key}.json"), "r") as f:
        copy_data = json.load(f)
        top_copy_heads = copy_data["top_heads"][:k_heads]
        
    with open(os.path.join(OUTPUT_DIR, f"head_entropy_{model_key}.json"), "r") as f:
        entropy_data = json.load(f)
        entropy_matrix = entropy_data["entropy_matrix"]
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    copy_heads_list = [(h["layer"], h["head"]) for h in top_copy_heads]
    null_heads_list = find_matched_null_heads(top_copy_heads, entropy_matrix, num_heads)
    
    print(f"Target Copy Heads: {copy_heads_list}")
    print(f"Matched Null Heads: {null_heads_list}")
    
    # 1. Baseline
    print("\nEvaluating Baseline...")
    acc_baseline = evaluate_accuracy(model, tokenizer, dataset)
    print(f"Baseline Exact Match: {acc_baseline*100:.1f}%")
    
    # 2. Ablate Null
    print("\nEvaluating Null Ablation...")
    hook_null = AblationHook(null_heads_list, num_heads, head_dim)
    hook_null.register(model)
    acc_null = evaluate_accuracy(model, tokenizer, dataset)
    hook_null.remove()
    print(f"Null Ablation Exact Match: {acc_null*100:.1f}%")
    
    # 3. Ablate Copy
    print("\nEvaluating Copy Ablation...")
    hook_copy = AblationHook(copy_heads_list, num_heads, head_dim)
    hook_copy.register(model)
    acc_copy = evaluate_accuracy(model, tokenizer, dataset)
    hook_copy.remove()
    print(f"Copy Ablation Exact Match: {acc_copy*100:.1f}%")
    
    results = {
        "model": model_key,
        "k_heads": k_heads,
        "baseline_acc": acc_baseline,
        "null_acc": acc_null,
        "copy_acc": acc_copy,
        "copy_heads": copy_heads_list,
        "null_heads": null_heads_list,
        "falsification_passed": (acc_null - acc_copy) > 0.10 # Must drop > 10% more than null
    }
    
    with open(os.path.join(OUTPUT_DIR, f"copy_ablation_{model_key}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results!")

if __name__ == "__main__":
    run_ablation("qwen-0.5b")
