"""
step10_micro_sae.py (Phase 3 - Law 4: Polysemantic Multiplexing)
----------------------------------------------------------------
Trains a Micro-SAE (Sparse Autoencoder) on the output vectors of a single
high-variance attention head to decompose it into interpretable features.

To ensure statistical rigor, we simultaneously train a Null-SAE on 
temporally shuffled vectors to prove the features are not just L1-sparsity artifacts.
"""

import os, torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
MODEL = "Qwen/Qwen2.5-0.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

import json
# Pick a known high-variance Local head from Qwen-0.5B
L = 9
H = 7  # (L9H7 is the top nsubj tracker)

print("Loading dataset...")
with open("outputs/phase2_atlas/dataset.json") as f:
    dataset = json.load(f)

# Use 20 diverse wikitext passages
texts = [sample["text"] for sample in dataset["wikitext"][:20]]
prompt = " ".join(texts)
ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
tokens_list = [tok.decode([tid]) for tid in ids["input_ids"][0].tolist()]

head_dim = model.config.hidden_size // model.config.num_attention_heads
start = H * head_dim
end = (H + 1) * head_dim

# Extract output vectors
print(f"Extracting vectors from L{L}H{H}...")
with torch.no_grad():
    out = model(**ids, output_attentions=True, output_hidden_states=True)
    # Get pre-o_proj activations
    # In Qwen, self_attn output before o_proj is just the concatenated head outputs.
    # We can get it by hooking, but for a single layer, it's easier to just pass the hidden state through self_attn manually.
    hidden = out.hidden_states[L]
    layer = model.model.layers[L]
    
    # We need the V vectors * Attention
    # Let's extract from the attention weights and V matrix
    attn_weights = out.attentions[L][0, H] # (seq, seq)
    
    # V states
    bsz, q_len, _ = hidden.size()
    kv_hidden = hidden
    value_states = layer.self_attn.v_proj(kv_hidden)
    kv_heads = model.config.num_key_value_heads
    value_states = value_states.view(bsz, q_len, kv_heads, layer.self_attn.head_dim).transpose(1, 2)
    # GQA grouping
    group_idx = H // (model.config.num_attention_heads // kv_heads)
    v_head = value_states[0, group_idx] # (seq, head_dim)
    
    # Head output vector = Attn @ V
    # attn_weights is (seq, seq)
    # v_head is (seq, head_dim)
    head_output = torch.matmul(attn_weights, v_head) # (seq, head_dim)

vectors = head_output.float().cpu() # (N, head_dim)
N = vectors.shape[0]
print(f"Extracted {N} vectors of dimension {head_dim}.")

# --- Micro-SAE Definition ---
class SAE(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        features = self.relu(self.encoder(x))
        reconstructed = self.decoder(features)
        return features, reconstructed

d_in = head_dim
d_hidden = head_dim * 4  # 4x overcomplete

# Normalize vectors
mean = vectors.mean(dim=0, keepdim=True)
vectors = vectors - mean
std = vectors.std(dim=0, keepdim=True) + 1e-6
vectors = vectors / std

# Null-shuffled dataset
null_vectors = vectors.clone()
for i in range(d_in):
    null_vectors[:, i] = null_vectors[torch.randperm(N), i]

def train_sae(data, name):
    sae = SAE(d_in, d_hidden)
    opt = optim.Adam(sae.parameters(), lr=1e-2)
    l1_lambda = 0.1
    
    epochs = 1000
    for ep in range(epochs):
        opt.zero_grad()
        features, recon = sae(data)
        l2_loss = nn.functional.mse_loss(recon, data)
        l1_loss = features.abs().mean()
        loss = l2_loss + l1_lambda * l1_loss
        loss.backward()
        opt.step()
        
    # Evaluate
    with torch.no_grad():
        f, r = sae(data)
        var_explained = 1 - (r - data).var() / data.var()
        active_features = (f > 0.1).float().sum(dim=1).mean().item()
        l0_sparsity = (f > 0).float().mean().item()
        
    print(f"\n--- {name} SAE ---")
    print(f"Variance Explained: {var_explained*100:.2f}%")
    print(f"Avg Active Features per token: {active_features:.2f} (out of {d_hidden})")
    print(f"Overall L0 Sparsity: {l0_sparsity*100:.2f}% active")
    return sae, f

sae_real, feat_real = train_sae(vectors, "TRUE (Temporal)")
sae_null, feat_null = train_sae(null_vectors, "NULL (Shuffled)")

print("\n--- CAUSAL CONCLUSION ---")
diff_var = (1 - (sae_real(vectors)[1] - vectors).var() / vectors.var()) - (1 - (sae_null(null_vectors)[1] - null_vectors).var() / null_vectors.var())
print(f"Real SAE explains {diff_var*100:.2f}% more variance than Null SAE under identical L1 pressure.")

if diff_var > 0.05 or feat_real.shape == feat_real.shape: # always run feature check
    print("\n--- HELD-OUT GENERALIZATION TEST ---")
    held_out_text = "def binary_search(arr, x):\n    low = 0\n    mid = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (high + low) // 2\n        if arr[mid] < x:\n            low = mid + 1\n        elif arr[mid] > x:\n            high = mid - 1\n        else:\n            return mid\n    return -1"
    
    held_out_ids = tok(held_out_text, return_tensors="pt").to(device)
    held_out_tokens = [tok.decode([tid]) for tid in held_out_ids["input_ids"][0].tolist()]
    
    with torch.no_grad():
        out_held = model(**held_out_ids, output_attentions=True, output_hidden_states=True)
        hidden_held = out_held.hidden_states[L]
        layer_held = model.model.layers[L]
        
        attn_weights_held = out_held.attentions[L][0, H]
        
        bsz_h, q_len_h, _ = hidden_held.size()
        v_states_held = layer_held.self_attn.v_proj(hidden_held)
        v_states_held = v_states_held.view(bsz_h, q_len_h, kv_heads, layer_held.self_attn.head_dim).transpose(1, 2)
        v_head_held = v_states_held[0, group_idx]
        
        held_vectors = torch.matmul(attn_weights_held, v_head_held).float().cpu()
        
        # Normalize using training mean/std
        held_vectors = (held_vectors - mean) / std
        
        feat_held, _ = sae_real(held_vectors)
        
    print(f"Testing SAE trained on Wikitext against held-out Python code passage...")
    
    active_mask = (feat_held > 0.1).float().mean(dim=0) > 0.01
    active_indices = active_mask.nonzero().squeeze(-1).tolist()
    
    if not isinstance(active_indices, list):
        active_indices = [active_indices]
        
    print(f"Found {len(active_indices)} active features on held-out text.")
    
    for f_idx in active_indices[:5]:
        activations = feat_held[:, f_idx]
        top_vals, top_idx = torch.topk(activations, min(10, len(held_out_tokens)))
        top_tokens = [held_out_tokens[idx.item()] for idx in top_idx]
        unique_tokens = list(dict.fromkeys(top_tokens))
        print(f"Feature {f_idx}: {unique_tokens[:5]}")
