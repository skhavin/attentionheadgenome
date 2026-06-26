import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import numpy as np

os.makedirs("outputs/phase8_paper_suite", exist_ok=True)
model_id = "Qwen/Qwen2.5-1.5B"

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, output_attentions=True)
model.eval()

clean_prompt = "The code for access is 4829. The code for denial is 1134. The code for access is"
corrupt_prompt = "The code for access is 9999. The code for denial is 1134. The code for access is"

clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(model.device)

print("Running clean pass for attention analysis...")
with torch.no_grad():
    clean_outputs = model(**clean_inputs)
    
expected_clean_id = tokenizer.encode(" 4829", add_special_tokens=False)[0]
expected_corrupt_id = tokenizer.encode(" 9999", add_special_tokens=False)[0]

attentions = clean_outputs.attentions 
seq_len = clean_inputs.input_ids.shape[1]

tokens = tokenizer.convert_ids_to_tokens(clean_inputs.input_ids[0])
prefix_idx = -1
copied_idx = -1
for i, t in enumerate(tokens):
    if "access" in t.lower() and i < len(tokens) - 5:
        prefix_idx = i + 1 
        copied_idx = prefix_idx + 1 
        break

# Print removed due to encoding issues
print(f"Prefix token index: {prefix_idx}, Copied token index: {copied_idx}")

head_scores = []
for layer_idx, layer_attn in enumerate(attentions):
    attn = layer_attn[0, :, -1, :] 
    for head_idx in range(attn.shape[0]):
        mass_prefix = attn[head_idx, prefix_idx].item()
        mass_copied = attn[head_idx, copied_idx].item()
        head_scores.append({
            "layer": layer_idx,
            "head": head_idx,
            "mass_prefix": mass_prefix,
            "mass_copied": mass_copied,
            "total_induction": mass_prefix + mass_copied
        })

head_scores.sort(key=lambda x: x["total_induction"], reverse=True)
induction_heads = head_scores[:40]

median_layer = np.median([h["layer"] for h in induction_heads])
early_heads = [h for h in induction_heads if h["layer"] <= median_layer]
late_heads = [h for h in induction_heads if h["layer"] > median_layer]

avg_early_prefix = np.mean([h["mass_prefix"] for h in early_heads])
avg_early_copied = np.mean([h["mass_copied"] for h in early_heads])
avg_late_prefix = np.mean([h["mass_prefix"] for h in late_heads])
avg_late_copied = np.mean([h["mass_copied"] for h in late_heads])

attention_analysis = {
    "early_heads_mass_prefix": float(avg_early_prefix),
    "early_heads_mass_copied": float(avg_early_copied),
    "late_heads_mass_prefix": float(avg_late_prefix),
    "late_heads_mass_copied": float(avg_late_copied)
}

print("Attention analysis complete.")
print(attention_analysis)

# Causal Patching
corrupt_cache = {}
def get_activation_hook(name):
    def hook(module, input, output):
        corrupt_cache[name] = output[0].detach()
    return hook

hooks_to_remove = []
for layer_idx in range(model.config.num_hidden_layers):
    attn_layer = model.model.layers[layer_idx].self_attn
    hooks_to_remove.append(attn_layer.q_proj.register_forward_hook(get_activation_hook(f"q_{layer_idx}")))
    hooks_to_remove.append(attn_layer.k_proj.register_forward_hook(get_activation_hook(f"k_{layer_idx}")))
    hooks_to_remove.append(attn_layer.v_proj.register_forward_hook(get_activation_hook(f"v_{layer_idx}")))

print("Extracting corrupt cache...")
with torch.no_grad():
    _ = model(**corrupt_inputs)

for h in hooks_to_remove: h.remove()

num_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
heads_per_kv = num_heads // num_kv_heads

def patch_hook(name, heads_to_patch, is_kv=False):
    def hook(module, input, output):
        patched_output = output[0].clone()
        corrupted_activation = corrupt_cache[name]
        for head_idx in heads_to_patch:
            if is_kv:
                target_idx = head_idx // heads_per_kv
            else:
                target_idx = head_idx
            start_idx = target_idx * head_dim
            end_idx = start_idx + head_dim
            patched_output[:, :, start_idx:end_idx] = corrupted_activation[:, :, start_idx:end_idx]
        return (patched_output,) + output[1:]
    return hook

def run_patching_experiment(patch_type, heads_by_layer):
    active_hooks = []
    for layer_idx, heads in heads_by_layer.items():
        if len(heads) == 0: continue
        attn_layer = model.model.layers[layer_idx].self_attn
        if patch_type == "QK":
            active_hooks.append(attn_layer.q_proj.register_forward_hook(patch_hook(f"q_{layer_idx}", heads, is_kv=False)))
            active_hooks.append(attn_layer.k_proj.register_forward_hook(patch_hook(f"k_{layer_idx}", heads, is_kv=True)))
        elif patch_type == "V":
            active_hooks.append(attn_layer.v_proj.register_forward_hook(patch_hook(f"v_{layer_idx}", heads, is_kv=True)))
            
    with torch.no_grad():
        out = model(**clean_inputs)
        
    for h in active_hooks: h.remove()
        
    logits = out.logits[0, -1, :]
    prob_clean = torch.softmax(logits, dim=-1)[expected_clean_id].item()
    prob_corrupt = torch.softmax(logits, dim=-1)[expected_corrupt_id].item()
    return float(prob_clean), float(prob_corrupt)

clean_logits = clean_outputs.logits[0, -1, :]
baseline_clean = torch.softmax(clean_logits, dim=-1)[expected_clean_id].item()
baseline_corrupt = torch.softmax(clean_logits, dim=-1)[expected_corrupt_id].item()

early_heads_dict = {i: [] for i in range(model.config.num_hidden_layers)}
for h in early_heads: early_heads_dict[h["layer"]].append(h["head"])

late_heads_dict = {i: [] for i in range(model.config.num_hidden_layers)}
for h in late_heads: late_heads_dict[h["layer"]].append(h["head"])

print("Running patching experiments...")
qk_early_clean, qk_early_corrupt = run_patching_experiment("QK", early_heads_dict)
v_late_clean, v_late_corrupt = run_patching_experiment("V", late_heads_dict)
qk_late_clean, qk_late_corrupt = run_patching_experiment("QK", late_heads_dict)
v_early_clean, v_early_corrupt = run_patching_experiment("V", early_heads_dict)

results = {
    "attention_analysis": attention_analysis,
    "baseline": {"clean_prob": float(baseline_clean), "corrupt_prob": float(baseline_corrupt)},
    "qk_patch_early": {"clean_prob": qk_early_clean, "corrupt_prob": qk_early_corrupt},
    "v_patch_late": {"clean_prob": v_late_clean, "corrupt_prob": v_late_corrupt},
    "qk_patch_late_control": {"clean_prob": qk_late_clean, "corrupt_prob": qk_late_corrupt},
    "v_patch_early_control": {"clean_prob": v_early_clean, "corrupt_prob": v_early_corrupt}
}

with open("outputs/phase8_paper_suite/causal_patching_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("SUCCESS")
