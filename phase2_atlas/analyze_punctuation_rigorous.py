import json
import os
import sys
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODELS = ["gpt2-medium", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "unsloth/Llama-3.2-1B"]
DATASET = "outputs/phase2_atlas/dataset.json"

print("=== RIGOROUS PUNCTUATION BASE RATE ANALYSIS ===")

with open(DATASET) as f:
    dataset = json.load(f)

for MODEL in MODELS:
    SAFE_MODEL = MODEL.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"MODEL: {SAFE_MODEL}")
    print(f"{'='*60}")
    
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        print(f"Failed to load tokenizer for {MODEL}: {e}")
        continue
        
    total_tokens = 0
    comma_tokens = 0
    period_tokens = 0
    
    # Run dataset through exact tokenizer logic used in step3
    for sample in dataset["ud_ewt"]:
        word_tokens = sample["tokens"]
        text = " ".join(word_tokens)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
        subword_ids = ids["input_ids"][0].tolist()
        
        # Count token occurrences natively
        total_tokens += len(subword_ids)
        for tid in subword_ids:
            decoded = tok.decode([tid]).strip()
            if decoded == ",":
                comma_tokens += 1
            elif decoded == ".":
                period_tokens += 1
                
    comma_base_rate = comma_tokens / total_tokens if total_tokens else 0
    period_base_rate = period_tokens / total_tokens if total_tokens else 0
    punct_base_rate = comma_base_rate + period_base_rate

    print(f"True Tokenized Base Rate (N={total_tokens}):")
    print(f"  Commas:  {comma_base_rate*100:.2f}%")
    print(f"  Periods: {period_base_rate*100:.2f}%")
    print(f"  Total Punct: {punct_base_rate*100:.2f}%")
    
    # Load atlas to find outlier heads
    path = f"outputs/phase2_atlas/{SAFE_MODEL}_head_atlas.json"
    if not os.path.exists(path):
        print("Atlas not found.")
        continue
        
    with open(path) as f:
        atlas = json.load(f)
        
    max_punct_mass = 0
    max_punct_head = None
    for key, h in atlas["heads"].items():
        if "grammar_profile" in h and "punct" in h["grammar_profile"]:
            val = h["grammar_profile"]["punct"]
            if val > max_punct_mass:
                max_punct_mass = val
                max_punct_head = h
                
    if max_punct_head:
        N_samples = total_tokens
        z_stat = (max_punct_mass - punct_base_rate) / np.sqrt((punct_base_rate * (1 - punct_base_rate)) / N_samples)
        p_val_z = stats.norm.sf(z_stat)
        
        print(f"\nTop Punctuation Head L{max_punct_head['layer']}H{max_punct_head['head']}:")
        print(f"  Mass allocated to punct: {max_punct_mass*100:.2f}%")
        print(f"  Z-Statistic (vs {punct_base_rate*100:.2f}% base): {z_stat:.2f}")
        print(f"  P-value: {p_val_z:.2e}")
