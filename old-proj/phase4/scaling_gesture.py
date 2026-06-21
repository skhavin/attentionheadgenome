# Quick check: do attention prototype clusters form on Qwen2.5-1.5B?
# We only run profiling (no full benchmark) — just enough to show clusters exist.
# This answers the reviewer question: "does this scale?"

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from config import DEVICE, USE_FP16, PHASE4_DIR, TOP_K_ATTENTION, MAX_SEQ_LEN, NUM_CLUSTERS

QWEN_1_5B = "Qwen/Qwen2.5-1.5B"

def main():
    print(f"Loading {QWEN_1_5B} (this needs ~3GB VRAM in fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_1_5B, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN_1_5B, trust_remote_code=True, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    # Only need ~50 docs to check if clusters form
    articles = load_articles(split="train", max_articles=50)

    # Extract patterns for just 50 docs
    all_patterns = []
    for text in tqdm(articles[:50], desc="Profiling Qwen-1.5B"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        seq_len = tokens["input_ids"].shape[1]
        if seq_len < 10:
            continue

        with torch.no_grad():
            output = model(**tokens, output_attentions=True)

        patterns = {}
        q_indices = torch.arange(seq_len).view(1, seq_len, 1)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)
        
        for li, la in enumerate(output.attentions):
            attn = la[0].float().cpu()
            attn = attn.masked_fill(mask, -float("inf"))
            k = min(TOP_K_ATTENTION, seq_len)
            _, top_indices = attn.topk(k, dim=2)
            rel_offsets = top_indices - q_indices
            
            for hi in range(attn.shape[0]):
                offsets = rel_offsets[hi].flatten().numpy()
                offsets = offsets[offsets <= 0]
                
                abs_off = np.abs(offsets)
                abs_off = np.minimum(abs_off, MAX_SEQ_LEN - 1)
                counts = np.bincount(abs_off, minlength=MAX_SEQ_LEN)[:MAX_SEQ_LEN]
                total = counts.sum()
                patterns[(li, hi)] = counts / total if total > 0 else counts.astype(float)
        all_patterns.append(patterns)

    # Cluster a few heads and report inertia
    keys = sorted(all_patterns[0].keys())[:10]  # just first 10 heads
    print(f"\nClustering results (first 10 heads, {len(all_patterns)} docs):")
    for layer, head in keys:
        data = np.array([d[(layer, head)] for d in all_patterns if (layer, head) in d])
        k = min(NUM_CLUSTERS, len(data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        print(f"  Layer {layer:2d}, Head {head:2d}: inertia={kmeans.inertia_:.4f} <-- {'CLUSTERS EXIST' if kmeans.inertia_ < 1.0 else 'weak clusters'}")

    print("\nDone! If inertia values are low, clusters form at 1.5B scale too.")

if __name__ == "__main__":
    main()
