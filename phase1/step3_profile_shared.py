# step3_profile_shared.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Shared profiling logic used by all step3_profile_*.py wrappers.
#          Import this module — do not run it directly.
#
# HOW IT WORKS:
#   1. Reads dataset_index.json to get the shared 300 article indices.
#   2. Loads each article, tokenizes, runs the model, extracts attention patterns.
#   3. For each (layer, head): computes a relative-offset histogram (length = SEQ_LEN).
#      Histogram[d] = fraction of top-k attended tokens that are exactly d positions back.
#   4. Saves raw histograms as pkl (fast) and mean summary as json (portable).
#
# OUTPUTS (written by the caller after calling profile()):
#   outputs/phase1/{slug}_patterns.pkl        — list of dicts: (layer,head) -> hist
#   outputs/phase1/{slug}_patterns_summary.json — per-head mean histogram

import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

# ── constants ─────────────────────────────────────────────────────────────────
SEQ_LEN = 512    # truncate all docs to this many tokens
TOP_K   = 10     # top-k attended positions to record per query token


def load_articles_from_index(index_path, num_docs=300):
    """Load exactly the articles listed in dataset_index.json."""
    with open(index_path) as f:
        cfg = json.load(f)

    ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])

    # Group rows into full articles
    all_articles = []
    current = []
    for row in ds:
        text = row["text"].strip()
        if text.startswith("= ") and text.endswith(" =") and text.count("=") == 2:
            if current:
                all_articles.append(" ".join(current))
            current = [text]
        elif text:
            current.append(text)
    if current:
        all_articles.append(" ".join(current))
    all_articles = [a for a in all_articles if len(a) > 100]

    indices  = cfg["indices"][:num_docs]
    selected = [all_articles[i] for i in indices if i < len(all_articles)]
    print(f"  Loaded {len(selected)} articles from shared index (first idx={indices[0]})")
    return selected


def extract_patterns_one_doc(model, tokenizer, text):
    """
    Forward one text through the model and extract per-(layer, head) histograms.
    Returns dict: (layer_idx, head_idx) -> np.ndarray shape (SEQ_LEN,) float32
    Returns None if the tokenized text is too short to be useful.
    """
    device = next(model.parameters()).device

    tokens  = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=SEQ_LEN
    )
    tokens  = {k: v.to(device) for k, v in tokens.items()}
    seq_len = tokens["input_ids"].shape[1]

    if seq_len < 10:
        return None

    # causal mask: True = upper-triangle (future) positions to mask out
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)
    q_idx = torch.arange(seq_len).view(1, seq_len, 1)   # (1, S, 1)

    with torch.no_grad():
        out = model(**tokens, output_attentions=True)

    patterns = {}
    for layer_idx, layer_attn in enumerate(out.attentions):
        if layer_attn is None:
            continue
        # Cast to float32 — 4-bit models return bfloat16 which np can't handle
        attn = layer_attn[0].float().cpu()               # (H, S, S)
        attn = attn.masked_fill(mask, -1e9)

        k = min(TOP_K, seq_len)
        _, top_idx = attn.topk(k, dim=2)                 # (H, S, k)
        rel_off = top_idx - q_idx                        # (H, S, k)

        for head_idx in range(attn.shape[0]):
            offs     = rel_off[head_idx].flatten().numpy()
            offs     = offs[offs <= 0]                   # keep only causal (past) positions
            abs_off  = np.abs(offs).astype(int)
            abs_off  = np.minimum(abs_off, SEQ_LEN - 1)
            counts   = np.bincount(abs_off, minlength=SEQ_LEN)[:SEQ_LEN].astype(np.float32)
            total    = counts.sum()
            patterns[(layer_idx, head_idx)] = counts / total if total > 0 else counts

    return patterns


def profile(model, tokenizer, index_path, num_docs, out_dir, model_slug):
    """
    Main profiling loop. Runs the model on num_docs articles and saves results.

    Args:
        model:       loaded HuggingFace model (eval mode)
        tokenizer:   matching tokenizer
        index_path:  path to dataset_index.json
        num_docs:    how many docs to use (max 300)
        out_dir:     where to write outputs
        model_slug:  short name used in filenames (e.g. "gpt2-medium")
    """
    os.makedirs(out_dir, exist_ok=True)

    texts = load_articles_from_index(index_path, num_docs)

    all_patterns = []
    for text in tqdm(texts, desc=f"  Profiling {model_slug}"):
        p = extract_patterns_one_doc(model, tokenizer, text)
        if p is not None:
            all_patterns.append(p)

    print(f"  Profiled {len(all_patterns)} / {len(texts)} docs successfully.")

    # ── save raw histograms (pkl) ──────────────────────────────────────────
    pkl_path = os.path.join(out_dir, f"{model_slug}_patterns.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_patterns, f)
    print(f"  Saved pkl -> {pkl_path}")

    # ── save mean-per-head summary (json) ─────────────────────────────────
    # JSON stores the mean histogram per (layer, head) as a plain float list.
    # Keys are strings "layer_head" because JSON requires string keys.
    if all_patterns:
        keys    = sorted(all_patterns[0].keys())
        summary = {}
        for layer, head in keys:
            hists = [d[(layer, head)] for d in all_patterns if (layer, head) in d]
            mean  = np.mean(hists, axis=0).tolist()   # list of 512 floats
            summary[f"{layer}_{head}"] = mean

        json_path = os.path.join(out_dir, f"{model_slug}_patterns_summary.json")
        with open(json_path, "w") as f:
            json.dump({"model_slug": model_slug, "num_docs": len(all_patterns),
                       "seq_len": SEQ_LEN, "heads": summary}, f)
        print(f"  Saved json -> {json_path}")

    return all_patterns
