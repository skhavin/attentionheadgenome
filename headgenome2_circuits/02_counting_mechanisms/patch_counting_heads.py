import json
import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase2_circuits"
DATASET_PATH = "headgenome2_circuits/datasets/counting.json"

class PatchingHook:
    def __init__(self, target_heads, num_heads, head_dim):
        self.target_heads = target_heads # list of (layer, head)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []
        self.cache = {}
        self.mode = "cache" # "cache" or "patch"
        
    def create_hook(self, layer_idx, target_heads_in_layer):
        def hook(module, input):
            x = input[0] # (batch, seq, hidden_dim)
            # For patching, we only care about the last token position
            # because that's where the answer is generated
            seq_len = x.shape[1]
            last_token = seq_len - 1
            
            for h in target_heads_in_layer:
                start_idx = h * self.head_dim
                end_idx = start_idx + self.head_dim
                
                if self.mode == "cache":
                    # Store the activation from the source prompt
                    self.cache[(layer_idx, h)] = x[:, last_token:, start_idx:end_idx].clone().detach()
                elif self.mode == "patch":
                    # Inject the cached activation into the target prompt
                    if (layer_idx, h) in self.cache:
                        # Ensure shapes align. Target might have length 1 if generating, 
                        # or we patch only the final prompt token.
                        x[:, -1:, start_idx:end_idx] = self.cache[(layer_idx, h)]
            return (x,)
        return hook

    def register(self, model):
        layer_to_heads = {}
        for l, h in self.target_heads:
            if l not in layer_to_heads:
                layer_to_heads[l] = []
            layer_to_heads[l].append(h)
            
        for layer_idx, target_heads_in_layer in layer_to_heads.items():
            layer_module = model.model.layers[layer_idx].self_attn.o_proj
            handle = layer_module.register_forward_pre_hook(self.create_hook(layer_idx, target_heads_in_layer))
            self.hooks.append(handle)
            
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.cache = {}

def get_integer_prediction(model, tokenizer, input_ids):
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=2,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Extract first integer
    import re
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return -1

def run_patching(model_key="qwen-0.5b", k_heads=4):
    print("Loading model and datasets...")
    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=False, output_hidden_states=False)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    with open(os.path.join(OUTPUT_DIR, f"counting_heads_{model_key}.json"), "r") as f:
        counting_data = json.load(f)
        top_heads = counting_data["top_heads"][:k_heads]
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    counting_heads_list = [(h["layer"], h["head"]) for h in top_heads]
    
    # Just for simplicity, we pick random null heads from the same layers
    null_heads_list = []
    for l, h in counting_heads_list:
        rand_h = (h + 1) % num_heads
        null_heads_list.append((l, rand_h))
        
    print(f"Target Counting Heads: {counting_heads_list}")
    print(f"Matched Null Heads: {null_heads_list}")
    
    # We need pairs of (Count=X, Count=X+2)
    # Let's organize dataset by count
    by_count = {}
    for item in dataset:
        c = item["count"]
        if c not in by_count: by_count[c] = []
        by_count[c].append(item)
        
    # Create pairs
    pairs = []
    for c in sorted(by_count.keys()):
        if c + 2 in by_count:
            # zip pairs
            for t_item, s_item in zip(by_count[c], by_count[c+2]):
                pairs.append((t_item, s_item)) # (target=X, source=X+2)
                
    if len(pairs) > 50:
        pairs = pairs[:50]
        
    print(f"Generated {len(pairs)} patching pairs (+2 shift tests).")
    
    results_baseline = []
    results_null_patch = []
    results_target_patch = []
    
    hook_null = PatchingHook(null_heads_list, num_heads, head_dim)
    hook_null.register(model)
    
    hook_target = PatchingHook(counting_heads_list, num_heads, head_dim)
    hook_target.register(model)
    
    # We will do this manually for each pair
    # To avoid hook conflicts, we manage modes carefully.
    
    for target_item, source_item in tqdm(pairs, desc="Patching"):
        target_ids = tokenizer.encode(target_item["prompt"], return_tensors="pt").to(model.device)
        source_ids = tokenizer.encode(source_item["prompt"], return_tensors="pt").to(model.device)
        
        # 1. Baseline Target
        base_pred = get_integer_prediction(model, tokenizer, target_ids)
        results_baseline.append(base_pred)
        
        # 2. Null Patch
        hook_null.mode = "cache"
        with torch.no_grad(): model(source_ids) # caches source null heads
        hook_null.mode = "patch"
        null_pred = get_integer_prediction(model, tokenizer, target_ids)
        results_null_patch.append(null_pred)
        hook_null.cache = {} # clear
        
        # 3. Target Patch
        hook_target.mode = "cache"
        with torch.no_grad(): model(source_ids) # caches source counting heads
        hook_target.mode = "patch"
        target_pred = get_integer_prediction(model, tokenizer, target_ids)
        results_target_patch.append(target_pred)
        hook_target.cache = {} # clear
        
    hook_null.remove()
    hook_target.remove()
    
    # Calculate shifts
    # We want to see if target_pred is closer to base_pred + 2 than null_pred is.
    null_shifts = [n - b for n, b in zip(results_null_patch, results_baseline) if b != -1 and n != -1]
    target_shifts = [t - b for t, b in zip(results_target_patch, results_baseline) if b != -1 and t != -1]
    
    print(f"Avg Null Shift: {np.mean(null_shifts):.3f}")
    print(f"Avg Target Shift: {np.mean(target_shifts):.3f}")
    
    # Wilcoxon signed-rank test between target_shifts and null_shifts
    # We need exactly paired data
    paired_null = []
    paired_target = []
    for b, n, t in zip(results_baseline, results_null_patch, results_target_patch):
        if b != -1 and n != -1 and t != -1:
            paired_null.append(n - b)
            paired_target.append(t - b)
            
    if len(paired_target) >= 10:
        stat, p_value = wilcoxon(paired_target, paired_null, alternative="greater")
    else:
        stat, p_value = 0, 1.0 # not enough data
        
    print(f"Wilcoxon p-value: {p_value:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, f"counting_patch_{model_key}.json"), "w") as f:
        json.dump({
            "model": model_key,
            "avg_null_shift": float(np.mean(paired_null)) if paired_null else 0.0,
            "avg_target_shift": float(np.mean(paired_target)) if paired_target else 0.0,
            "wilcoxon_p": float(p_value),
            "falsification_passed": bool(p_value < 0.05)
        }, f, indent=2)

if __name__ == "__main__":
    run_patching("qwen-0.5b")
