"""
02_extract_full_genome.py

The heavy-lifting extraction script. Runs on the original 4 canonical architectures
(GPT-2, Qwen-0.5B, Qwen-1.5B, Llama-3.2-1B) and extracts the full Bin-1 + Bin-2
feature set for every single attention head.

Key design decisions:
- Uses a HELD-OUT prompt set (never seen by Phase 1 label derivation)
- Merges extracted features with canonical labels from outputs/canonical_labels.json
- Enforces the Bin-3 leak assertion before writing any output
- Run with --debug to test on a dummy 2-layer model before committing to full extraction

Usage:
    python 02_extract_full_genome.py --debug      # Quick smoke-test
    python 02_extract_full_genome.py              # Full extraction (~2-4 hours)
"""

import sys
import os
import json
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import our schema — single source of truth
sys.path.insert(0, os.path.dirname(__file__))
from triage_and_schema import (
    extract_bin1_features, assert_no_bin3_leak,
    APPROVED_FEATURES, BIN_3_FEATURES, compute_gini, compute_entropy
)

# ============================================================
# CONFIG
# ============================================================

CANONICAL_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "canonical_labels.json"
)

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "full_genome_dataset.csv")

# These are the exact 4 architectures from Table 1.
# Do NOT add new architectures here without first running the Phase 1
# entropy-collapse labeling procedure on them.
MODELS = [
    {
        "model_id": "gpt2",
        "model_name": "GPT-2",
        "label_key": "gpt2",
    },
    {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "model_name": "Qwen-0.5B",
        "label_key": "qwen_0.5b",
    },
    {
        "model_id": "Qwen/Qwen2.5-1.5B",
        "model_name": "Qwen-1.5B",
        "label_key": "qwen_1.5b",
    },
    {
        "model_id": "meta-llama/Llama-3.2-1B",
        "model_name": "Llama-3.2-1B",
        "label_key": "llama_3.2_1b",
    },
]

# Held-out prompt config
# We use a fresh slice of wikitext that was never used in Phase 1
WIKITEXT_SPLIT = "test"
NUM_PROMPTS = 50
PROMPT_SEQ_LEN = 512  # tokens per prompt

# NIAH prompts: synthesize fresh ones (different needles from Phase 1)
NIAH_NEEDLES = [
    ("The secret activation code is", "ZEPHYR-7749"),
    ("The hidden passkey is", "CRIMSON-4421"),
    ("The override phrase is", "NOVA-DELTA-09"),
    ("The vault combination is", "8831-SIGMA"),
    ("The launch sequence is", "TANAGER-LIMA-6"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HELD-OUT DATASET BUILDER
# ============================================================

def build_held_out_prompts(tokenizer, n=NUM_PROMPTS, seq_len=PROMPT_SEQ_LEN):
    """
    Builds the held-out prompt set from the wikitext TEST split.
    This is a different split from what Phase 1 used (validation).
    Returns a list of token tensors.
    """
    print(f"  Loading wikitext-2 {WIKITEXT_SPLIT} split for held-out prompts...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=WIKITEXT_SPLIT)

    full_text = " ".join(ds["text"])
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    prompts = []
    stride = len(tokens) // n
    for i in range(n):
        chunk = tokens[i * stride: i * stride + seq_len]
        if len(chunk) == seq_len:
            prompts.append(torch.tensor(chunk).unsqueeze(0))
    
    print(f"  Built {len(prompts)} held-out prompts of length {seq_len}.")
    return prompts


def build_niah_prompts(tokenizer, context_len=512):
    """
    Synthesizes fresh NIAH prompts with different needle/haystack combos.
    """
    prompts = []
    labels = []
    
    filler = "The weather was pleasant and the birds sang softly in the trees. " * 50
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    
    for prefix_text, answer in NIAH_NEEDLES:
        # Embed needle at random position in filler
        full_text = f"{filler[:200]} {prefix_text} {answer}. {filler[200:]}"
        tokens = tokenizer.encode(full_text, add_special_tokens=False)[:context_len]
        prompts.append(torch.tensor(tokens).unsqueeze(0))
        
        # Find needle position
        needle_tokens = tokenizer.encode(answer, add_special_tokens=False)
        labels.append({"needle_tokens": needle_tokens, "answer": answer})
    
    return prompts, labels


# ============================================================
# HOOK-BASED ACTIVATION CAPTURE
# ============================================================

class AttentionCapture:
    """
    Registers forward hooks on each attention layer to capture
    per-head attention patterns and pre-softmax scores.
    """
    def __init__(self):
        self.hooks = []
        self.captured = {}  # layer_idx -> {"attn_weights": tensor, "qk_scores": tensor}

    def register(self, model):
        """Auto-detect attention layers and register hooks."""
        for name, module in model.named_modules():
            if hasattr(module, 'attn') or "attention" in name.lower():
                # We hook the module's forward to intercept attention weights
                pass
        # Simpler approach: hook at the attention output level
        # This works across GPT-2, Qwen, Llama architectures
        self._register_output_hooks(model)

    def _register_output_hooks(self, model):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, attn_weights, ...)
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.captured[layer_idx] = {
                            "attn_weights": attn_weights.detach().cpu()
                        }
            return hook_fn

        layer_idx = 0
        for name, module in model.named_modules():
            # Works for GPT2Attention, Qwen2Attention, LlamaAttention
            if any(t in type(module).__name__ for t in ["Attention", "attention"]):
                if hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                    h = module.register_forward_hook(make_hook(layer_idx))
                    self.hooks.append(h)
                    layer_idx += 1

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================
# BIN-2 FEATURE EXTRACTION FROM ACTIVATIONS
# ============================================================

def extract_bin2_features_from_attn(attn_weights, layer_idx, head_idx):
    """
    Extract Bin-2 behavioral features from a captured attention weight tensor.

    attn_weights: [batch, n_heads, seq_len, seq_len]
    """
    feats = {}

    # Get per-head attention pattern [seq_len, seq_len]
    if attn_weights.ndim == 4:
        head_attn = attn_weights[0, head_idx].numpy()  # [seq_len, seq_len]
    else:
        return feats  # Unexpected shape, skip

    seq_len = head_attn.shape[-1]

    # Softmax attention distribution for the last query token (decode-like behavior)
    last_attn = head_attn[-1]  # [seq_len]
    total = last_attn.sum() + 1e-10

    # Attention entropy
    feats["attention_entropy_mean"] = float(compute_entropy(last_attn))

    # Gini
    feats["attention_gini"] = float(compute_gini(last_attn))

    # Top-k mass
    sorted_attn = np.sort(last_attn)[::-1]
    feats["attention_top1_mass"] = float(sorted_attn[0] / total)
    feats["attention_top5_mass"] = float(sorted_attn[:5].sum() / total)

    # BOS and first-token mass
    feats["bos_mass"] = float(last_attn[0] / total)
    feats["first_4_token_mass"] = float(last_attn[:4].sum() / total)

    # Positional distance (how far back does this head look?)
    positions = np.arange(seq_len)
    distances = (seq_len - 1) - positions  # distance from last token
    attn_normalized = last_attn / total

    feats["mean_attention_distance"] = float(np.sum(attn_normalized * distances))
    feats["median_attention_distance"] = float(
        positions[np.searchsorted(np.cumsum(attn_normalized[::-1])[::-1], 0.5)]
        if seq_len > 0 else 0
    )

    # Local mass at various windows
    for window in [32, 64, 128, 256]:
        local_idx = max(0, seq_len - window)
        feats[f"local_mass_{window}"] = float(last_attn[local_idx:].sum() / total)

    # Long-range mass at various windows
    for window in [512, 1024]:
        far_idx = max(0, seq_len - window)
        feats[f"long_range_mass_{window}"] = float(last_attn[:far_idx].sum() / total)

    # Distance decay slope (log-linear fit)
    try:
        bin_edges = np.array([1, 4, 8, 16, 32, 64, 128, 256, seq_len])
        bin_edges = bin_edges[bin_edges <= seq_len]
        bin_masses = []
        for i in range(len(bin_edges) - 1):
            lo = seq_len - bin_edges[i + 1]
            hi = seq_len - bin_edges[i]
            bin_masses.append(last_attn[lo:hi].sum() / total)
        
        if len(bin_masses) > 2:
            log_dist = np.log(bin_edges[1:len(bin_masses)+1] + 1)
            log_mass = np.log(np.array(bin_masses) + 1e-10)
            slope = np.polyfit(log_dist, log_mass, 1)[0]
            feats["distance_decay_slope"] = float(slope)
        else:
            feats["distance_decay_slope"] = 0.0
    except Exception:
        feats["distance_decay_slope"] = 0.0

    # QK score approximation from attention weights (pre-softmax estimate via log)
    # NOTE: We don't have direct access to raw QK scores via the output hook,
    # so we approximate via log(attention_weights) + log(Z) where Z is unknown.
    # This gives us relative ordering but not absolute magnitudes.
    # True pre-softmax scores require a deeper intervention (separate hook on QK).
    log_attn = np.log(np.maximum(last_attn, 1e-10))
    feats["qk_score_mean"] = float(log_attn.mean())
    feats["qk_score_std"] = float(log_attn.std())
    feats["qk_score_max"] = float(log_attn.max())
    feats["qk_score_min"] = float(log_attn.min())
    feats["qk_top1_minus_top2"] = float(
        np.sort(log_attn)[-1] - np.sort(log_attn)[-2] if len(log_attn) > 1 else 0.0
    )
    feats["qk_top1_zscore"] = float(
        (log_attn.max() - log_attn.mean()) / (log_attn.std() + 1e-10)
    )
    feats["qk_score_gini"] = float(compute_gini(log_attn))
    feats["qk_score_kurtosis"] = float(
        float(np.mean((log_attn - log_attn.mean()) ** 4) / (log_attn.std() ** 4 + 1e-10))
    )

    # Noise floor: fraction of tokens with log-attention score > mean + 2*std
    # This directly measures the Phase 2 false-spike problem
    threshold = log_attn.mean() + 2 * log_attn.std()
    spikes = log_attn > threshold
    feats["qk_score_noise_floor"] = float(spikes.mean())
    feats["qk_false_spike_rate"] = float(spikes.mean())

    # First hit distance: how far back is the first spike (approx early-exit behavior)
    spike_positions = np.where(spikes)[0]
    if len(spike_positions) > 0:
        feats["first_hit_distance"] = float(seq_len - spike_positions[-1])
    else:
        feats["first_hit_distance"] = float(seq_len)

    # Placeholders for features that need token-type info (filled during NIAH pass)
    feats["proper_noun_mass"] = 0.0
    feats["number_mass"] = 0.0
    feats["rare_token_mass"] = 0.0
    feats["function_word_mass"] = 0.0
    feats["content_word_mass"] = 0.0
    feats["true_target_rank_by_qk_score"] = 0.0
    feats["true_target_score_percentile"] = 0.0
    feats["punctuation_mass"] = float(last_attn[:4].sum() / total)  # proxy
    feats["delimiter_mass"] = 0.0

    return feats


# ============================================================
# PER-MODEL WEIGHT EXTRACTION
# ============================================================

def get_head_weights(model, layer_idx, head_idx, n_heads, n_kv_heads, head_dim):
    """
    Extract W_Q, W_K, W_V, W_O slices for a specific head.
    Handles GPT-2 (c_attn fused) and Qwen/Llama (separate q_proj/k_proj/v_proj).
    Returns (W_Q, W_K, W_V, W_O) each as 2D tensors.
    """
    # Get the attention module for this layer
    layer = None
    attn_module = None

    # Try different model architectures
    for name, module in model.named_modules():
        if f"layers.{layer_idx}" in name or f"h.{layer_idx}" in name:
            if any(t in type(module).__name__ for t in ["Attention", "attention"]):
                if hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                    attn_module = module
                    break

    if attn_module is None:
        return None, None, None, None

    # GPT-2 style: fused c_attn [3 * hidden, hidden]
    if hasattr(attn_module, "c_attn"):
        W_fused = attn_module.c_attn.weight.detach()  # [3*hidden, hidden]
        hidden = W_fused.shape[1]
        W_Q_full = W_fused[:hidden]
        W_K_full = W_fused[hidden:2*hidden]
        W_V_full = W_fused[2*hidden:]

        # Slice out head
        start = head_idx * head_dim
        end = start + head_dim
        W_Q = W_Q_full[start:end]
        W_K = W_K_full[start:end]
        W_V = W_V_full[start:end]
        W_O = attn_module.c_proj.weight.detach()[:, start:end].T

    # Qwen/Llama style: separate q_proj, k_proj, v_proj
    elif hasattr(attn_module, "q_proj"):
        W_Q_full = attn_module.q_proj.weight.detach()  # [n_heads * head_dim, hidden]
        W_K_full = attn_module.k_proj.weight.detach()  # [n_kv_heads * head_dim, hidden]
        W_V_full = attn_module.v_proj.weight.detach()  # [n_kv_heads * head_dim, hidden]
        W_O_full = attn_module.o_proj.weight.detach()  # [hidden, n_heads * head_dim]

        # GQA: map query head -> KV head
        kv_head_idx = head_idx // (n_heads // n_kv_heads)

        q_start = head_idx * head_dim
        q_end = q_start + head_dim
        kv_start = kv_head_idx * head_dim
        kv_end = kv_start + head_dim

        W_Q = W_Q_full[q_start:q_end]
        W_K = W_K_full[kv_start:kv_end]
        W_V = W_V_full[kv_start:kv_end]
        W_O = W_O_full[:, q_start:q_end].T

    else:
        return None, None, None, None

    return W_Q, W_K, W_V, W_O


# ============================================================
# MAIN EXTRACTION LOOP
# ============================================================

def run_extraction(debug=False):
    """
    Main loop: for each model, extract all Bin-1 and Bin-2 features
    for every attention head and save to CSV.
    """
    # Load canonical labels
    print(f"\nLoading canonical labels from: {CANONICAL_LABELS_PATH}")
    if not os.path.exists(CANONICAL_LABELS_PATH):
        print(f"  [ERROR] canonical_labels.json not found at {CANONICAL_LABELS_PATH}")
        print("  Run Phase 1 labeling first.")
        return

    with open(CANONICAL_LABELS_PATH, "r") as f:
        canonical_labels = json.load(f)

    all_rows = []

    models_to_run = MODELS[:1] if debug else MODELS  # Debug: only GPT-2
    
    for model_config in models_to_run:
        model_id = model_config["model_id"]
        model_name = model_config["model_name"]
        label_key = model_config["label_key"]
        
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if debug:
            # Use tiny dummy model config for fast smoke-test
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            config.num_hidden_layers = 2
            model = AutoModelForCausalLM.from_config(config).to(DEVICE)
            print("  [DEBUG] Loaded 2-layer dummy model.")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=DEVICE
            )

        model.eval()

        # Architecture metadata
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
        head_dim = model.config.hidden_size // n_heads

        print(f"  Layers: {n_layers}, Heads: {n_heads}, KV Heads: {n_kv_heads}, Head Dim: {head_dim}")

        # Build held-out prompts
        held_out_prompts = build_held_out_prompts(tokenizer, n=5 if debug else NUM_PROMPTS)

        # Set up attention capture hooks
        capture = AttentionCapture()
        capture.register(model)

        # Aggregate Bin-2 features across all held-out prompts
        # Shape: layer_idx -> head_idx -> list of feature dicts
        accumulated_bin2 = {
            l: {h: [] for h in range(n_heads)} for l in range(n_layers)
        }

        print(f"  Running {len(held_out_prompts)} held-out prompts...")
        for i, prompt_tokens in enumerate(held_out_prompts):
            capture.clear()
            
            prompt_tokens = prompt_tokens.to(DEVICE)

            with torch.no_grad():
                _ = model(prompt_tokens, output_attentions=True)

            # For each layer where we captured attention weights
            for layer_idx, data in capture.captured.items():
                attn_weights = data["attn_weights"]
                if attn_weights is not None:
                    for head_idx in range(min(n_heads, attn_weights.shape[1])):
                        feats = extract_bin2_features_from_attn(
                            attn_weights, layer_idx, head_idx
                        )
                        accumulated_bin2[layer_idx][head_idx].append(feats)

            if (i + 1) % 10 == 0:
                print(f"    Prompts processed: {i+1}/{len(held_out_prompts)}")

        capture.remove()

        # Now extract Bin-1 features and combine with averaged Bin-2 features
        print(f"  Extracting Bin-1 static weight features...")
        model_labels = canonical_labels.get(label_key, {})

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                row = {
                    "model_name": model_name,
                    "model_id": model_id,
                }

                # Get label
                label_key_head = f"L{layer_idx}H{head_idx}"
                row["canonical_label"] = model_labels.get(label_key_head, "unknown")

                # Skip heads with no label in debug mode
                if debug and row["canonical_label"] == "unknown":
                    continue

                # Bin-1: static weight features
                W_Q, W_K, W_V, W_O = get_head_weights(
                    model, layer_idx, head_idx, n_heads, n_kv_heads, head_dim
                )

                if W_Q is not None:
                    bin1_feats = extract_bin1_features(
                        W_Q, W_K, W_V, W_O,
                        layer_idx, head_idx, n_layers, n_heads, n_kv_heads
                    )
                    row.update(bin1_feats)

                # Bin-2: average behavioral features across all prompts
                head_bin2_samples = accumulated_bin2.get(layer_idx, {}).get(head_idx, [])
                if head_bin2_samples:
                    avg_feats = {}
                    all_keys = head_bin2_samples[0].keys()
                    for k in all_keys:
                        values = [s[k] for s in head_bin2_samples if k in s]
                        avg_feats[k] = float(np.mean(values)) if values else 0.0
                    row.update(avg_feats)

                all_rows.append(row)

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if not all_rows:
        print("\n[ERROR] No rows collected. Check canonical_labels.json format.")
        return

    df = pd.DataFrame(all_rows)

    # MANDATORY: Assert no Bin-3 features leaked in
    assert_no_bin3_leak(df)
    print("\n[OK] Bin-3 leak assertion passed.")

    # NaN report
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) > 0:
        print(f"\n[WARN] NaN values found in {len(nan_cols)} columns:")
        print(nan_cols)
    else:
        print("[OK] No NaN values in dataset.")

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(df)} rows x {len(df.columns)} columns to:")
    print(f"  {OUTPUT_CSV}")
    print(f"\nLabel distribution:")
    print(df["canonical_label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Run on 2-layer dummy GPT-2 only for smoke-testing.")
    args = parser.parse_args()

    run_extraction(debug=args.debug)
