# step2_quant_check.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Quantization sanity check — profile Qwen-1.5B in BF16 and in 4-bit on
#          the same 50 docs, cluster with k=4, and compute Jaccard similarity
#          of cluster assignments between the two runs.
#
# PASS CRITERION: Jaccard >= 0.95 means 4-bit quantization is safe for all models.
# FAIL CRITERION: Jaccard < 0.95 means Llama-8B 4-bit data may be distorted.
#
# OUTPUTS:
#   outputs/phase1/quant_check_bf16_labels.pkl
#   outputs/phase1/quant_check_4bit_labels.pkl
#   outputs/phase1/quant_check_result.json
#
# IMPORTANT: This script must finish before step3 profiling of Llama-8B begins.
# IMPORTANT: BF16 Qwen-1.5B needs ~4GB VRAM. 4-bit needs ~1.5GB.

import os
import sys
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH   = os.path.join(OUT_DIR, "dataset_index.json")
OUT_BF16_PKL = os.path.join(OUT_DIR, "quant_check_bf16_labels.pkl")
OUT_4BIT_PKL = os.path.join(OUT_DIR, "quant_check_4bit_labels.pkl")
OUT_JSON     = os.path.join(OUT_DIR, "quant_check_result.json")

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-1.5B"
NUM_DOCS   = 50     # short run — only for the quant check
SEQ_LEN    = 512    # truncate docs to this many tokens
TOP_K      = 10     # top-k attended positions to histogram
K_CLUSTERS = 4

HF_CACHE = "d:\\.cache\\huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["SAFETENSORS_FAST_GPU"] = "1"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_articles_from_index(index_path, num_docs):
    """Load only the articles at the positions listed in dataset_index.json."""
    with open(index_path) as f:
        cfg = json.load(f)

    ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])

    # Group all rows into articles (same logic as old-proj/data_utils.py)
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

    # Pick exactly the shared indices, capped at num_docs
    indices = cfg["indices"][:num_docs]
    selected = [all_articles[i] for i in indices if i < len(all_articles)]
    print(f"Loaded {len(selected)} articles from shared index.")
    return selected


def extract_histograms(model, tokenizer, texts):
    """Run model on each text, return list of dicts: (layer, head) -> histogram."""
    device = next(model.parameters()).device
    mask_upper = None   # built lazily when we know seq_len

    all_patterns = []
    for text in tqdm(texts, desc="  Profiling"):
        tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=SEQ_LEN
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        seq_len = tokens["input_ids"].shape[1]

        if seq_len < 10:
            continue

        # Rebuild causal mask if sequence length changed
        if mask_upper is None or mask_upper.shape[0] != seq_len:
            mask_upper = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)

        with torch.no_grad():
            out = model(**tokens, output_attentions=True)

        q_idx = torch.arange(seq_len).view(1, seq_len, 1)   # (1, S, 1)
        patterns = {}

        for layer_idx, layer_attn in enumerate(out.attentions):
            if layer_attn is None:
                continue
            # layer_attn: (batch=1, heads, seq, seq) — cast to float32 for stability
            attn = layer_attn[0].float().cpu()              # (H, S, S)
            attn = attn.masked_fill(mask_upper, -1e9)       # mask future tokens

            k = min(TOP_K, seq_len)
            _, top_idx = attn.topk(k, dim=2)                # (H, S, k)
            rel_off = top_idx - q_idx                       # (H, S, k)

            for head_idx in range(attn.shape[0]):
                offs = rel_off[head_idx].flatten().numpy()  # all (S*k) offsets
                offs = offs[offs <= 0]                      # causal: only attend backwards
                abs_off = np.abs(offs).astype(int)
                abs_off = np.minimum(abs_off, SEQ_LEN - 1)
                counts = np.bincount(abs_off, minlength=SEQ_LEN)[:SEQ_LEN].astype(float)
                total = counts.sum()
                patterns[(layer_idx, head_idx)] = counts / total if total > 0 else counts

        all_patterns.append(patterns)

    return all_patterns


def get_head_means(all_patterns):
    """Compute mean histogram per head."""
    if not all_patterns:
        return {}
    keys = sorted(all_patterns[0].keys())
    head_means = {}
    for layer, head in keys:
        hists = [d[(layer, head)] for d in all_patterns if (layer, head) in d]
        head_means[(layer, head)] = np.mean(hists, axis=0)
    return head_means


def load_model(quantize):
    """Load Qwen-1.5B in BF16 (quantize=False) or 4-bit (quantize=True)."""
    print(f"\nLoading {MODEL_ID}  quantize={quantize} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()
    return model, tokenizer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] dataset_index.json not found at {INDEX_PATH}")
        print("        Run step1_generate_index.py first.")
        sys.exit(1)

    texts = load_articles_from_index(INDEX_PATH, NUM_DOCS)

    # 1. Profile BF16
    model, tok = load_model(quantize=False)
    patterns_bf16 = extract_histograms(model, tok, texts)
    del model
    torch.cuda.empty_cache()

    # 2. Profile 4-bit
    model, tok = load_model(quantize=True)
    patterns_4bit = extract_histograms(model, tok, texts)
    del model
    torch.cuda.empty_cache()

    # 3. Compute mean histograms per head
    head_means_bf16 = get_head_means(patterns_bf16)
    head_means_4bit = get_head_means(patterns_4bit)

    keys = sorted(head_means_bf16.keys())
    X_bf16 = np.array([head_means_bf16[k] for k in keys])
    X_4bit = np.array([head_means_4bit[k] for k in keys])

    # 4. Fit KMeans on BF16
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels_bf16 = km.fit_predict(X_bf16)

    # 5. Predict on 4-bit using BF16 centroids
    labels_4bit = km.predict(X_4bit)

    # Save labels mapping
    bf16_label_dict = {keys[i]: int(labels_bf16[i]) for i in range(len(keys))}
    fourbit_label_dict = {keys[i]: int(labels_4bit[i]) for i in range(len(keys))}

    with open(OUT_BF16_PKL, "wb") as f:
        pickle.dump(bf16_label_dict, f)
    with open(OUT_4BIT_PKL, "wb") as f:
        pickle.dump(fourbit_label_dict, f)

    # Compute similarity (fraction of heads that fall in the same cluster)
    j = float(np.mean(labels_bf16 == labels_4bit))
    verdict = "PASS" if j >= 0.95 else "FAIL"
    note = (
        "4-bit quantization does not distort cluster assignments. Llama-8B 4-bit data is valid."
        if verdict == "PASS"
        else "4-bit distorts attention signatures. Re-profile Llama-8B in mixed precision."
    )

    out = {
        "model":             MODEL_ID,
        "num_docs":          len(texts),
        "k_clusters":        K_CLUSTERS,
        "jaccard_similarity": round(j, 4),
        "threshold":         0.95,
        "verdict":           verdict,
        "note":              note,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n--- Quantization Check Result ---")
    print(f"  Jaccard similarity : {j:.4f}")
    print(f"  Verdict            : {verdict}")
    print(f"  Note               : {note}")
    print(f"  Saved to           : {OUT_JSON}")


if __name__ == "__main__":
    main()
