import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys

os.makedirs("outputs/phase8_paper_suite", exist_ok=True)

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
model.eval()

# 1. Identify Hyper-Diagonal Heads dynamically
print("Computing W_OV diagonal ratios...")
head_ratios = []

if "gpt2" in model_id.lower():
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    n_embd = model.config.n_embd
    for layer_idx in range(num_layers):
        c_attn_w = model.transformer.h[layer_idx].attn.c_attn.weight
        c_proj_w = model.transformer.h[layer_idx].attn.c_proj.weight
        w_v_all = c_attn_w[:, 2*n_embd : 3*n_embd]
        for head_idx in range(num_heads):
            start = head_idx * head_dim
            end = start + head_dim
            w_v_head = w_v_all[:, start:end]
            w_o_head = c_proj_w[start:end, :]
            w_ov = w_v_head @ w_o_head
            diag = torch.diagonal(w_ov)
            diag_sum = diag.abs().sum().item()
            off_diag_sum = w_ov.abs().sum().item() - diag_sum
            ratio = (diag_sum / n_embd) / (off_diag_sum / (n_embd * (n_embd - 1))) if off_diag_sum > 0 else 0
            head_ratios.append({"layer": layer_idx, "head": head_idx, "ratio": ratio})
else:
    # Qwen or Llama
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    n_embd = model.config.hidden_size
    num_queries_per_kv = num_heads // num_kv_heads
    for layer_idx in range(num_layers):
        attn = model.model.layers[layer_idx].self_attn
        v_proj = attn.v_proj.weight
        o_proj = attn.o_proj.weight
        for head_idx in range(num_heads):
            kv_idx = head_idx // num_queries_per_kv
            w_v_head = v_proj[kv_idx*head_dim : (kv_idx+1)*head_dim, :].t()
            w_o_head = o_proj[:, head_idx*head_dim : (head_idx+1)*head_dim]
            w_ov = w_v_head @ w_o_head.t()
            diag = torch.diagonal(w_ov)
            diag_sum = diag.abs().sum().item()
            off_diag_sum = w_ov.abs().sum().item() - diag_sum
            ratio = (diag_sum / n_embd) / (off_diag_sum / (n_embd * (n_embd - 1))) if off_diag_sum > 0 else 0
            head_ratios.append({"layer": layer_idx, "head": head_idx, "ratio": ratio})

# Get top 15 hyper-diagonal heads
head_ratios.sort(key=lambda x: x["ratio"], reverse=True)
top_heads = head_ratios[:15]
hyper_diagonal_heads = {i: [] for i in range(num_layers)}
for h in top_heads:
    hyper_diagonal_heads[h["layer"]].append(h["head"])

print(f"Ablating top 15 hyper-diagonal heads. Min ratio: {top_heads[-1]['ratio']:.2f}")

# 2. Datasets
dataset_exact = [
    ("The UUID is 9f86d081884c. The UUID is", " 9f86d081884c"),
    ("The verification hash is A7X9Q2. The verification hash is", " A7X9Q2"),
    ("My license plate is XYZ-9921. My license plate is", " XYZ-9921"),
    ("The token ID is b4k9m1. The token ID is", " b4k9m1")
]

dataset_semantic = [
    ("The capital of France is Paris. The capital of France is", " Paris"),
    ("The color of the sky is blue. The color of the sky is", " blue"),
    ("The opposite of hot is cold. The opposite of hot is", " cold"),
    ("A dog is an animal. A dog is an", " animal")
]

def evaluate(prompts, ablation=False):
    correct = 0
    hooks = []
    if ablation:
        def get_ablation_hook(heads_to_ablate):
            def hook(module, input):
                x = input[0].clone()
                for h in heads_to_ablate:
                    x[:, :, h*head_dim : (h+1)*head_dim] = 0.0
                return (x,)
            return hook
            
        for layer_idx, heads in hyper_diagonal_heads.items():
            if len(heads) > 0:
                if "gpt2" in model_id.lower():
                    hooks.append(model.transformer.h[layer_idx].attn.c_proj.register_forward_pre_hook(get_ablation_hook(heads)))
                else:
                    hooks.append(model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(get_ablation_hook(heads)))
    
    with torch.no_grad():
        for prompt, expected in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            expected_id = tokenizer.encode(expected)[0] if "gpt2" in model_id.lower() else tokenizer.encode(expected, add_special_tokens=False)[0]
            
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            pred_id = logits.argmax().item()
            
            if pred_id == expected_id:
                correct += 1
                
    for hook in hooks: hook.remove()
    return correct / len(prompts)

print("Evaluating...")
exact_base = evaluate(dataset_exact, ablation=False)
exact_abl = evaluate(dataset_exact, ablation=True)
sem_base = evaluate(dataset_semantic, ablation=False)
sem_abl = evaluate(dataset_semantic, ablation=True)

results = {
    "model": model_id,
    "top_hyper_diagonal_ratio_min": top_heads[-1]["ratio"],
    "exact_copy_accuracy": {"baseline": exact_base, "ablated": exact_abl},
    "semantic_copy_accuracy": {"baseline": sem_base, "ablated": sem_abl}
}

print(json.dumps(results, indent=2))
with open(f"outputs/phase8_paper_suite/hyper_diagonal_{model_id.replace('/', '_')}.json", "w") as f:
    json.dump(results, f, indent=2)
