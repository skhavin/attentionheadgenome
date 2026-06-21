# Repeat Phase 1 (attention profiling) on Qwen2.5-0.5B.
# Shows whether prototype clusters form regardless of architecture.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from config import (QWEN_MODEL_NAME, DEVICE, USE_FP16, PHASE4_DIR,
                    NUM_PROFILING_DOCS, TOP_K_ATTENTION, MAX_SEQ_LEN)

def extract_patterns(model, tokenizer, text):
    """Same as Phase 1 but for Qwen architecture."""
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
        attn = layer_attn[0].float().cpu()
        attn = attn.masked_fill(mask, -float("inf"))
        
        k = min(TOP_K_ATTENTION, seq_len)
        _, top_indices = attn.topk(k, dim=2)
        rel_offsets = top_indices - q_indices
        
        for head_idx in range(attn.shape[0]):
            offsets = rel_offsets[head_idx].flatten().numpy()
            offsets = offsets[offsets <= 0]
            
            abs_off = np.abs(offsets)
            abs_off = np.minimum(abs_off, MAX_SEQ_LEN - 1)
            counts = np.bincount(abs_off, minlength=MAX_SEQ_LEN)[:MAX_SEQ_LEN]
            
            total = counts.sum()
            patterns[(layer_idx, head_idx)] = counts / total if total > 0 else counts.astype(float)

    return patterns

def main():
    os.makedirs(PHASE4_DIR, exist_ok=True)

    print(f"Loading {QWEN_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    articles = load_articles(split="train", max_articles=NUM_PROFILING_DOCS)

    all_patterns = []
    for text in tqdm(articles, desc="Profiling Qwen"):
        p = extract_patterns(model, tokenizer, text)
        if p is not None:
            all_patterns.append(p)

    save_path = os.path.join(PHASE4_DIR, "qwen_attention_patterns.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_patterns, f)
    print(f"Saved {len(all_patterns)} Qwen patterns to {save_path}")

if __name__ == "__main__":
    main()
