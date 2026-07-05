"""
step8_causal_patching.py (Phase 3 - Law 2: Causal Patching)
-----------------------------------------------------------
This script causally ablates all known Retrieval Heads simultaneously
to test if they act as a distributed boolean AND-gate for a downstream Induction Head.

Methodology:
1. Run a Needle-In-A-Haystack prompt on Qwen2.5-1.5B.
2. Measure the Induction Head's attention to the needle.
3. Measure the final logit probability of the correct needle token.
4. Ablate ALL Retrieval Heads by zeroing out their slices in `o_proj.weight`.
5. Re-run and measure the exact mathematical drop.
"""

import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
MODEL = "Qwen/Qwen2.5-1.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

# From Phase 2 Atlas Qwen2.5-1.5B
RETRIEVAL_HEADS = [(0, 9), (5, 5), (11, 6), (11, 11), (12, 5), (26, 11)]

INDUCTION_LAYER = 21
INDUCTION_HEAD = 8

# Synthetic Needle-In-A-Haystack Prompt
prompt = "The secret color is BLUE. The weather is nice. I went to the store. Later, he asked for it. The secret color is"
# Tokenize
ids = tok(prompt, return_tensors="pt").to(device)
input_ids = ids["input_ids"][0].tolist()

target_str = " BLUE"
target_token_id = tok.encode(target_str, add_special_tokens=False)[0]

needle_pos = -1
for i, tid in enumerate(input_ids):
    if tid == target_token_id:
        needle_pos = i
        break

if needle_pos == -1:
    print("Error: Could not find needle token in prompt.")
    sys.exit(1)

print(f"Needle token found at position {needle_pos}: {tok.decode([target_token_id])}")

# ---------------------------------------------------------
# RUN 1: BASELINE
# ---------------------------------------------------------
with torch.no_grad():
    out_base = model(**ids, output_attentions=True)

# Induction head attention from last token to needle
attn_base = out_base.attentions[INDUCTION_LAYER][0, INDUCTION_HEAD, -1, needle_pos].item()

# Logit prob for BLUE
logits = out_base.logits[0, -1, :]
probs = torch.softmax(logits, dim=-1)
prob_base = probs[target_token_id].item()
rank_base = (probs > prob_base).sum().item() + 1

print("\n--- BASELINE ---")
print(f"Induction Head L{INDUCTION_LAYER}H{INDUCTION_HEAD} Attention to Needle: {attn_base*100:.2f}%")
print(f"Logit Probability of 'BLUE': {prob_base*100:.2f}% (Rank: {rank_base})")

# ---------------------------------------------------------
# RUN 2: ABLATION (MULTIPLE HEADS)
# ---------------------------------------------------------
print(f"\n--- ABLATING ALL RETRIEVAL HEADS: {RETRIEVAL_HEADS} ---")
head_dim = model.config.hidden_size // model.config.num_attention_heads

# Store original weights so we can restore them later
orig_weights = {}

for L, H in RETRIEVAL_HEADS:
    start = H * head_dim
    end = (H + 1) * head_dim
    weight = model.model.layers[L].self_attn.o_proj.weight.data
    
    if L not in orig_weights:
        orig_weights[L] = weight.clone()
        
    weight[:, start:end] = 0.0

with torch.no_grad():
    out_abl = model(**ids, output_attentions=True)

# Restore weights
for L, w in orig_weights.items():
    model.model.layers[L].self_attn.o_proj.weight.data.copy_(w)

# Induction head attention from last token to needle
attn_abl = out_abl.attentions[INDUCTION_LAYER][0, INDUCTION_HEAD, -1, needle_pos].item()

# Logit prob for BLUE
logits_abl = out_abl.logits[0, -1, :]
probs_abl = torch.softmax(logits_abl, dim=-1)
prob_abl = probs_abl[target_token_id].item()
rank_abl = (probs_abl > prob_abl).sum().item() + 1

print(f"Induction Head L{INDUCTION_LAYER}H{INDUCTION_HEAD} Attention to Needle: {attn_abl*100:.2f}%")
print(f"Logit Probability of 'BLUE': {prob_abl*100:.2f}% (Rank: {rank_abl})")

print("\n--- CAUSAL EFFECT ---")
attn_drop = attn_base - attn_abl
prob_drop = prob_base - prob_abl
print(f"Attention Drop: {attn_drop*100:.2f}%")
print(f"Probability Drop: {prob_drop*100:.2f}%")
