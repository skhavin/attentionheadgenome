import json
import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

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
        self.has_printed_sanity = False
        
    def create_hook(self, layer_idx, target_heads_in_layer):
        def hook(module, input):
            x = input[0] # (batch, seq, hidden_dim)
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
                            # Verify patch took effect
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

class MLPTractingHook:
    def __init__(self, target_mlps):
        self.target_mlps = target_mlps
        self.hooks = []
        self.activations = {}
        
    def create_hook(self, layer_idx):
        def hook(module, input, output):
            # output of the MLP
            # Actually we want the input to the MLP to compute fused pre-act shift.
            pass
        return hook
        
    def create_pre_hook(self, layer_idx, gate_proj_weight, up_proj_weight):
        def hook(module, input):
            x = input[0] # (batch, seq, hidden_dim)
            last_tok_x = x[:, -1:, :]
            # Compute pre-act: (x @ W_gate^T) * (x @ W_up^T)
            gate = torch.matmul(last_tok_x, gate_proj_weight.T)
            up = torch.matmul(last_tok_x, up_proj_weight.T)
            fused_pre_act = gate * up
            self.activations[layer_idx] = fused_pre_act.detach().clone()
            return (x,)
        return hook

    def register(self, model):
        for layer_idx in self.target_mlps:
            mlp = model.model.layers[layer_idx].mlp
            w_gate = mlp.gate_proj.weight.detach()
            w_up = mlp.up_proj.weight.detach()
            handle = mlp.register_forward_pre_hook(self.create_pre_hook(layer_idx, w_gate, w_up))
            self.hooks.append(handle)
            
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

def run_mlp_patching(model_key="qwen-0.5b", k_heads=4):
    print("Loading model and datasets...")
    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=False, output_hidden_states=False)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    with open(os.path.join(OUTPUT_DIR, f"counting_heads_{model_key}.json"), "r") as f:
        top_heads = json.load(f)["top_heads"][:k_heads]
        
    with open(os.path.join(OUTPUT_DIR, f"frobenius_norms_{model_key}.json"), "r") as f:
        frobenius_data = json.load(f)["rankings"]
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    
    target_heads = [(h["layer"], h["head"]) for h in top_heads]
    target_layer = target_heads[0][0] # They are all L16 usually
    
    # Select 5 Null Heads from the same layer for noise floor calculation
    null_head_groups = []
    for offset in [1, 2, 3, 4, 5]:
        group = []
        for l, h in target_heads:
            group.append((l, (h + offset) % num_heads))
        null_head_groups.append(group)
        
    downstream_mlps = list(range(target_layer + 1, num_layers))
    
    # Dataset pairing (Count=X vs Count=X+2)
    by_count = {}
    for item in dataset:
        c = item["count"]
        if c not in by_count: by_count[c] = []
        by_count[c].append(item)
        
    pairs = []
    for c in sorted(by_count.keys()):
        if c + 2 in by_count:
            for t_item, s_item in zip(by_count[c], by_count[c+2]):
                pairs.append((t_item, s_item))
    pairs = pairs[:50] # Use 50 pairs
    
    # Data structures to track MLP shifts
    # shifts[mlp_layer] = list of float shift norms
    target_shifts = {mlp: [] for mlp in downstream_mlps}
    null_shifts = {null_idx: {mlp: [] for mlp in downstream_mlps} for null_idx in range(len(null_head_groups))}
    
    mlp_hook = MLPTractingHook(downstream_mlps)
    mlp_hook.register(model)
    
    for target_item, source_item in tqdm(pairs, desc="Patching MLPs"):
        target_ids = tokenizer.encode(target_item["prompt"], return_tensors="pt").to(model.device)
        source_ids = tokenizer.encode(source_item["prompt"], return_tensors="pt").to(model.device)
        
        # 1. Baseline Target
        with torch.no_grad(): model(target_ids)
        base_acts = {k: v.clone() for k, v in mlp_hook.activations.items()}
        mlp_hook.activations.clear()
        
        # 2. Target Patch
        hook_target = PatchingHook(target_heads, num_heads, head_dim)
        hook_target.register(model)
        hook_target.mode = "cache"
        with torch.no_grad(): model(source_ids)
        hook_target.mode = "patch"
        with torch.no_grad(): model(target_ids)
        target_acts = {k: v.clone() for k, v in mlp_hook.activations.items()}
        hook_target.remove()
        mlp_hook.activations.clear()
        
        for mlp in downstream_mlps:
            shift = torch.linalg.norm(target_acts[mlp] - base_acts[mlp]).item()
            target_shifts[mlp].append(shift)
            
        # 3. Null Patches
        for null_idx, null_group in enumerate(null_head_groups):
            hook_null = PatchingHook(null_group, num_heads, head_dim)
            hook_null.register(model)
            hook_null.mode = "cache"
            with torch.no_grad(): model(source_ids)
            hook_null.mode = "patch"
            with torch.no_grad(): model(target_ids)
            null_acts = {k: v.clone() for k, v in mlp_hook.activations.items()}
            hook_null.remove()
            mlp_hook.activations.clear()
            
            for mlp in downstream_mlps:
                shift = torch.linalg.norm(null_acts[mlp] - base_acts[mlp]).item()
                null_shifts[null_idx][mlp].append(shift)
                
    mlp_hook.remove()
    
    # Aggregate Frobenius Norms
    # frobenius_data is a list of dicts. We want sum of norms per MLP across the counting heads
    mlp_frob = {mlp: 0.0 for mlp in downstream_mlps}
    for item in frobenius_data:
        mlp = item["target_mlp"]
        if mlp in mlp_frob:
            mlp_frob[mlp] += item["norm_fused"]
            
    # Evaluation
    # 1. Correlation Variance
    frob_list = []
    target_shift_list = []
    for mlp in downstream_mlps:
        frob_list.append(mlp_frob[mlp])
        target_shift_list.append(np.mean(target_shifts[mlp]))
        
    r_val, p_val = pearsonr(frob_list, target_shift_list)
    
    # 2. Empirical Effect Size (Top Ranked MLP)
    top_mlp = max(mlp_frob.keys(), key=lambda x: mlp_frob[x])
    top_mlp_target_shift = np.mean(target_shifts[top_mlp])
    
    top_mlp_null_shifts = []
    for null_idx in range(len(null_head_groups)):
        top_mlp_null_shifts.append(np.mean(null_shifts[null_idx][top_mlp]))
        
    null_mean = np.mean(top_mlp_null_shifts)
    null_std = np.std(top_mlp_null_shifts)
    
    # Falsification Thresholds
    threshold_passed = top_mlp_target_shift > (null_mean + 2 * null_std)
    correlation_passed = (r_val > 0.30) and (p_val < 0.01)
    
    results = {
        "model": model_key,
        "falsification_passed": bool(threshold_passed and correlation_passed),
        "correlation": {
            "r_value": float(r_val),
            "p_value": float(p_val),
            "passed": bool(correlation_passed)
        },
        "empirical_effect": {
            "top_mlp": top_mlp,
            "target_shift": float(top_mlp_target_shift),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "threshold": float(null_mean + 2 * null_std),
            "passed": bool(threshold_passed)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, f"mlp_routing_{model_key}.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Correlation: r={r_val:.3f}, p={p_val:.4f}")
    print(f"Top MLP ({top_mlp}) Shift: {top_mlp_target_shift:.3f}")
    print(f"Null Distribution: mu={null_mean:.3f}, std={null_std:.3f}")
    print(f"Falsification Passed: {results['falsification_passed']}")

if __name__ == "__main__":
    run_mlp_patching("qwen-0.5b")
