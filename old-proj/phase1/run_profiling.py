# Run GPT-2 Medium on 500 WikiText-103 articles and extract attention patterns.
# For each (layer, head), record relative positions of top-k attended tokens.
# Output: attention_patterns.pkl in outputs/phase1/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from config import (MODEL_NAME, DEVICE, USE_FP16, PHASE1_DIR,
                    NUM_PROFILING_DOCS, TOP_K_ATTENTION, MAX_SEQ_LEN)

def extract_patterns_from_doc(model, tokenizer, text):
    """Run model on one article, return dict of (layer, head) -> relative position histogram."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    seq_len = tokens["input_ids"].shape[1]

    if seq_len < 10:
        return None

    with torch.no_grad():
        output = model(**tokens, output_attentions=True)

    patterns = {}
    q_indices = torch.arange(seq_len).view(1, seq_len, 1)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)

    for layer_idx, layer_attn in enumerate(output.attentions):
        attn = layer_attn[0].float().cpu()  # (heads, seq, seq)
        attn = attn.masked_fill(mask, -float("inf"))
        
        k = min(TOP_K_ATTENTION, seq_len)
        _, top_indices = attn.topk(k, dim=2)  # (heads, seq, k)
        rel_offsets = top_indices - q_indices  # (heads, seq, k)

        for head_idx in range(attn.shape[0]):
            offsets = rel_offsets[head_idx].flatten().numpy()
            offsets = offsets[offsets <= 0]  # keep valid causal attendances
            
            abs_off = np.abs(offsets)
            abs_off = np.minimum(abs_off, MAX_SEQ_LEN - 1)
            counts = np.bincount(abs_off, minlength=MAX_SEQ_LEN)[:MAX_SEQ_LEN]
            
            total = counts.sum()
            patterns[(layer_idx, head_idx)] = counts / total if total > 0 else counts.astype(float)

    return patterns

def main():
    os.makedirs(PHASE1_DIR, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    # Load 500 properly-grouped Wikipedia articles (train split for profiling)
    articles = load_articles(split="train", max_articles=NUM_PROFILING_DOCS)

    all_patterns = []
    for text in tqdm(articles, desc="Profiling"):
        p = extract_patterns_from_doc(model, tokenizer, text)
        if p is not None:
            all_patterns.append(p)

    save_path = os.path.join(PHASE1_DIR, "attention_patterns.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_patterns, f)
    print(f"Saved {len(all_patterns)} patterns to {save_path}")

if __name__ == "__main__":
    main()
