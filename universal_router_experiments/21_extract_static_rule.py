import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import gc

model_id = "Qwen/Qwen2.5-0.5B"
print(f"Loading {model_id} for Static Rule Extraction...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_heads

print("\n--- Step 1: 1-Shot 1000-Token Ground Truth Probe (1% Threshold) ---")

haystack_sentence = "The study of artificial intelligence has progressed rapidly over the past decade. "
needle_sentence = "The secret password to unlock the HeadGenome matrix is Triton. "

text = (haystack_sentence * 40) + needle_sentence + (haystack_sentence * 40) + "The secret password to unlock the HeadGenome matrix is"
inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

# Find the character index of "Triton"
char_idx = text.find("Triton")
needle_idx = -1

# Find which token spans this character index
offsets = inputs.offset_mapping[0]
for i, (start, end) in enumerate(offsets):
    if start <= char_idx < end:
        needle_idx = i
        break

# Remove offset_mapping before passing to model
inputs.pop("offset_mapping")
inputs = inputs.to("cuda")

if needle_idx == -1:
    print("Could not find Triton token sequence via offsets!")
    needle_idx = 500
    
print(f"Needle token index: {needle_idx} ('{tokenizer.decode(inputs.input_ids[0, needle_idx])}') out of {inputs.input_ids.shape[1]} tokens")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

ground_truth_retrieval_heads = set()
for layer_idx, attn_matrix in enumerate(outputs.attentions):
    final_token_attn = attn_matrix[0, :, -1, :] 
    for head_idx in range(n_heads):
        needle_mass = final_token_attn[head_idx, needle_idx].item()
        if needle_mass > 0.01: # 1% threshold
            ground_truth_retrieval_heads.add((layer_idx, head_idx))

print(f"Found {len(ground_truth_retrieval_heads)} Ground-Truth Retrieval/Multi-Hop Heads at 1% threshold.")

print("\n--- Step 2: Extracting Static Features & Synthesizing the Rule ---")

static_features = []
embed_matrix = model.get_input_embeddings().weight.detach()

for layer_idx in range(n_layers):
    q_proj = model.model.layers[layer_idx].self_attn.q_proj.weight.detach()
    k_proj = model.model.layers[layer_idx].self_attn.k_proj.weight.detach()
    v_proj = model.model.layers[layer_idx].self_attn.v_proj.weight.detach()
    
    q_proj = q_proj.view(n_heads, head_dim, -1)
    k_proj = k_proj.view(num_kv_heads, head_dim, -1)
    v_proj = v_proj.view(num_kv_heads, head_dim, -1)
    
    heads_per_kv = n_heads // num_kv_heads
    
    for head_idx in range(n_heads):
        is_retrieval = (layer_idx, head_idx) in ground_truth_retrieval_heads
        
        q_w = q_proj[head_idx]
        kv_idx = head_idx // heads_per_kv
        k_w = k_proj[kv_idx]
        v_w = v_proj[kv_idx]
        
        depth_ratio = layer_idx / n_layers
        
        q_norm = torch.norm(q_w).item()
        v_norm = torch.norm(v_w).item()
        vq_ratio = v_norm / q_norm if q_norm > 0 else 0
        
        k_embed = F.linear(embed_matrix, k_w)
        k_embed_norm = torch.norm(k_embed).item()
        
        k_baseline_norm = torch.norm(k_w).item() * torch.norm(embed_matrix).item()
        embed_k_lock = k_embed_norm / k_baseline_norm if k_baseline_norm > 0 else 0
        
        static_features.append({
            "layer": layer_idx,
            "head": head_idx,
            "is_retrieval": is_retrieval,
            "depth_ratio": depth_ratio,
            "vq_ratio": vq_ratio,
            "embed_k_lock": embed_k_lock
        })

print("\n--- Ground Truth Analysis ---")
retrieval_embed_locks = [f["embed_k_lock"] for f in static_features if f["is_retrieval"]]
retrieval_vq_ratios = [f["vq_ratio"] for f in static_features if f["is_retrieval"]]

if len(retrieval_embed_locks) > 0:
    print(f"Retrieval Heads - Mean Embed-K-Lock: {sum(retrieval_embed_locks)/len(retrieval_embed_locks):.3f}")
    print(f"Retrieval Heads - Mean V/Q Ratio: {sum(retrieval_vq_ratios)/len(retrieval_vq_ratios):.3f}")
else:
    print("No retrieval heads found!")

print("\n--- Proposed Universal Static Rule ---")
print('''
def is_retrieval_head(layer_idx, n_layers, q_w, k_w, v_w, embed_matrix):
    v_norm = torch.norm(v_w).item()
    q_norm = torch.norm(q_w).item()
    vq_ratio = v_norm / q_norm
    return vq_ratio > 1.25 and (layer_idx / n_layers) > 0.2
''')

with open("static_features_qwen0.5.json", "w") as f:
    json.dump(static_features, f)

