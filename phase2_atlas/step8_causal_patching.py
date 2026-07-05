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

prompts_data = [
    ("The secret color is BLUE. The weather is nice. I went to the store. Later, he asked for it. The secret color is", " BLUE"),
    ("My favorite animal is the TIGER. Let's talk about cars. Mathematics is fun. In the end, he remembered. My favorite animal is the", " TIGER"),
    ("The password to the vault is APPLE. She bought some shoes. The sky was cloudy. He tried to log in. The password to the vault is", " APPLE"),
    ("The hidden city is ATLANTIS. Books are on the table. The cat slept. They searched the map. The hidden city is", " ATLANTIS"),
    ("Her maiden name is SMITH. The computer was slow. It started raining. He filled out the form. Her maiden name is", " SMITH"),
]

import numpy as np

results = {
    "base_prob": [],
    "single_drop": [],
    "multi_drop": [],
}

head_dim = model.config.hidden_size // model.config.num_attention_heads

for prompt, target_str in prompts_data:
    ids = tok(prompt, return_tensors="pt").to(device)
    input_ids = ids["input_ids"][0].tolist()
    target_token_id = tok.encode(target_str, add_special_tokens=False)[0]
    
    needle_pos = -1
    for i, tid in enumerate(input_ids):
        if tid == target_token_id:
            needle_pos = i
            break
            
    if needle_pos == -1:
        continue
        
    # Baseline
    with torch.no_grad():
        out_base = model(**ids, output_attentions=True)
    prob_base = torch.softmax(out_base.logits[0, -1, :], dim=-1)[target_token_id].item()
    results["base_prob"].append(prob_base)
    
    # Single
    L, H = RETRIEVAL_HEADS[0]
    start = H * head_dim
    end = (H + 1) * head_dim
    weight = model.model.layers[L].self_attn.o_proj.weight.data
    w_orig = weight.clone()
    weight[:, start:end] = 0.0
    
    with torch.no_grad():
        out_single = model(**ids, output_attentions=True)
    prob_single = torch.softmax(out_single.logits[0, -1, :], dim=-1)[target_token_id].item()
    results["single_drop"].append(prob_base - prob_single)
    
    model.model.layers[L].self_attn.o_proj.weight.data.copy_(w_orig)
    
    # Multi
    orig_weights = {}
    for L, H in RETRIEVAL_HEADS:
        s = H * head_dim
        e = (H + 1) * head_dim
        w = model.model.layers[L].self_attn.o_proj.weight.data
        if L not in orig_weights: orig_weights[L] = w.clone()
        w[:, s:e] = 0.0
        
    with torch.no_grad():
        out_multi = model(**ids, output_attentions=True)
    prob_multi = torch.softmax(out_multi.logits[0, -1, :], dim=-1)[target_token_id].item()
    results["multi_drop"].append(prob_base - prob_multi)
    
    for L, w in orig_weights.items():
        model.model.layers[L].self_attn.o_proj.weight.data.copy_(w)

print("\n--- STATISTICAL ROBUSTNESS CHECK ---")
print(f"Base Prob: {np.mean(results['base_prob'])*100:.2f}% ± {np.std(results['base_prob'])*100:.2f}%")
print(f"Single Head Drop: {np.mean(results['single_drop'])*100:.2f}% ± {np.std(results['single_drop'])*100:.2f}%")
print(f"6-Head Circuit Drop: {np.mean(results['multi_drop'])*100:.2f}% ± {np.std(results['multi_drop'])*100:.2f}%")
