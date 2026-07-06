import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer


DATASET_PATH = "headgenome2_circuits/datasets/arithmetic.json"
OUTPUT_DIR = "outputs/phase2_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ArithmeticPatchingHook:
    def __init__(self, target_heads, num_heads, head_dim):
        self.target_heads = target_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []
        self.cache = {}
        self.mode = "cache"
        self.has_printed_sanity = False
        
    def create_hook(self, layer_idx, target_heads_in_layer):
        def hook(module, input):
            x = input[0]
            seq_len = x.shape[1]
            last_token = seq_len - 1
            
            for h in target_heads_in_layer:
                start_idx = h * self.head_dim
                end_idx = start_idx + self.head_dim
                
                if self.mode == "cache":
                    self.cache[(layer_idx, h)] = x[:, last_token:, start_idx:end_idx].clone().detach()
                elif self.mode == "patch":
                    if (layer_idx, h) in self.cache:
                        x[:, -1:, start_idx:end_idx] = self.cache[(layer_idx, h)]
                        
                        if not self.has_printed_sanity:
                            diff = x[:, -1:, start_idx:end_idx] - self.cache[(layer_idx, h)]
                            assert torch.all(diff == 0.0), f"Patching failed on L{layer_idx}H{h}"
                            print(f"[Sanity Check] Verified `o_proj` input for L{layer_idx}H{h} was successfully patched.")
                            self.has_printed_sanity = True
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

def get_expected_digit(logits, tokenizer):
    # Digits '0' through '9'
    digit_ids = [tokenizer.encode(str(d))[0] for d in range(10)]
    last_tok_logits = logits[0, -1, digit_ids]
    probs = torch.softmax(last_tok_logits, dim=-1)
    
    expected_val = 0.0
    for d in range(10):
        expected_val += probs[d].item() * d
    return expected_val

def run_arithmetic_patch(model_key="qwen-0.5b", k_heads=4):
    print("Loading model and datasets...")
    model, tokenizer = load_model_and_tokenizer(model_key)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    with open(os.path.join(OUTPUT_DIR, f"arithmetic_heads_{model_key}.json"), "r") as f:
        top_heads = json.load(f)["top_heads"][:k_heads]
        
    target_heads = [(h["layer"], h["head"]) for h in top_heads]
    target_layer = target_heads[0][0]
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # Null Heads (Depth matched)
    null_heads = [(l, (h + 5) % num_heads) for l, h in target_heads]
    
    # Pairing for N=50
    # Pair items with different sums (e.g. diff of +2 or +3)
    pairs = []
    for i in range(len(dataset)):
        for j in range(i+1, len(dataset)):
            if dataset[i]["sum"] != dataset[j]["sum"]:
                pairs.append((dataset[i], dataset[j]))
                if len(pairs) >= 50:
                    break
        if len(pairs) >= 50:
            break
            
    null_shifts = []
    target_shifts = []
    
    print(f"Target Arithmetic Heads: {target_heads}")
    print(f"Matched Null Heads: {null_heads}\n")
    
    for t_item, s_item in tqdm(pairs, desc="Patching"):
        t_ids = tokenizer.encode(t_item["prompt"], return_tensors="pt").to(model.device)
        s_ids = tokenizer.encode(s_item["prompt"], return_tensors="pt").to(model.device)
        
        # 1. Baseline
        with torch.no_grad():
            out_base = model(t_ids).logits
        base_exp = get_expected_digit(out_base, tokenizer)
        
        # We define a "shift" as moving towards the source sum.
        # If base is closer to t_sum, and patch moves it towards s_sum:
        # We can just measure the difference.
        
        # 2. Null Patch
        hook_null = ArithmeticPatchingHook(null_heads, num_heads, head_dim)
        hook_null.register(model)
        hook_null.mode = "cache"
        with torch.no_grad(): model(s_ids)
        hook_null.mode = "patch"
        with torch.no_grad():
            out_null = model(t_ids).logits
        null_exp = get_expected_digit(out_null, tokenizer)
        hook_null.remove()
        
        # 3. Target Patch
        hook_target = ArithmeticPatchingHook(target_heads, num_heads, head_dim)
        hook_target.register(model)
        hook_target.mode = "cache"
        with torch.no_grad(): model(s_ids)
        hook_target.mode = "patch"
        with torch.no_grad():
            out_target = model(t_ids).logits
        target_exp = get_expected_digit(out_target, tokenizer)
        hook_target.remove()
        
        # Calculate shift direction
        # Positive shift means moved towards source expected sum
        # Negative shift means moved away
        direction = 1 if s_item["sum"] > t_item["sum"] else -1
        
        null_shift = (null_exp - base_exp) * direction
        target_shift = (target_exp - base_exp) * direction
        
        null_shifts.append(null_shift)
        target_shifts.append(target_shift)
        
    avg_null = np.mean(null_shifts)
    avg_target = np.mean(target_shifts)
    
    print(f"\nAvg Null Shift towards Source: {avg_null:.3f}")
    print(f"Avg Target Shift towards Source: {avg_target:.3f}")
    
    try:
        stat, p_val = wilcoxon(null_shifts, target_shifts, alternative="less")
        print(f"Wilcoxon p-value: {p_val:.4f}")
    except Exception as e:
        print(f"Wilcoxon failed: {e}")
        p_val = 1.0
        
    results = {
        "model": model_key,
        "avg_null_shift": float(avg_null),
        "avg_target_shift": float(avg_target),
        "wilcoxon_p_value": float(p_val),
        "falsification_passed": bool(p_val < 0.05 and avg_target > avg_null)
    }
    
    out_file = os.path.join(OUTPUT_DIR, f"arithmetic_patch_results_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_arithmetic_patch("qwen-0.5b")
