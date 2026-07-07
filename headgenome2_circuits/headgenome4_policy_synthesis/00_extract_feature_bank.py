"""
00_extract_feature_bank.py
HeadGenome 2 — Phase 0: Feature Bank Extraction

Builds a single, flat CSV of structural + behavioral features for all 1,568
canonical heads across all 4 architectures. No forward-pass is needed for
most features; they come from existing head_atlas JSONs. Weight-derived features
(OV norm, effective ranks, singular spectra) require a brief model weight load
(no GPU forward pass, no tokenization).

Feature notes
-------------
QK features (qk_eff_rank, qk_sv*): computed via full SVD of W_Q @ W_K^T,
  the (head_dim, head_dim) weight-product matrix. This is a STATIC, weight-only
  quantity (capacity for content-matching), NOT a behavioural feature computed
  from empirical attention patterns on tokens. For a 64-dim head, W_Q @ W_K^T is
  a 64x64 matrix; full SVD is O(64^3) = trivial, no sampling is needed.

GQA group features: Instead of a raw group ID (which would leak architecture
  identity into Model B — Llama vs Qwen have different ID ranges), we use:
    gqa_group_size       = number of query heads sharing one KV head (= n_q / n_kv)
    gqa_within_group_rank = this head's index within its KV group (0..group_size-1)
  These are architecture-agnostic: a model with 8 query heads per KV group gets
  gqa_group_size=8 regardless of which model it is.

Missing atlas tolerance: see MISSING_ATLAS_ABORT_THRESHOLD below.

Output: outputs/phase0/feature_bank.csv
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "headgenome2_circuits" / "utils"))

CANONICAL_LABELS_PATH = ROOT / "outputs" / "canonical_labels.json"
ATLAS_DIR = ROOT / "outputs" / "phase2_atlas"
REGIME_DIR = ROOT / "outputs" / "phase8_paper_suite"
OUTPUT_DIR = ROOT / "outputs" / "phase0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fallback tolerance: if more than this fraction of heads are missing from the
# atlas (and therefore get NaN for all behavioral features), abort — the
# measurement inconsistency between labels and features is too large to trust.
# Flag-but-proceed below 5%; abort above.
MISSING_ATLAS_ABORT_THRESHOLD = 0.05   # 5% of total heads
MISSING_ATLAS_WARN_THRESHOLD  = 0.01   # 1%

# Map canonical label keys -> atlas file names
ATLAS_FILES = {
    "GPT-2":        ATLAS_DIR / "gpt2-medium_head_atlas.json",
    "Qwen-0.5B":    ATLAS_DIR / "Qwen2.5-0.5B_head_atlas.json",
    "Qwen-1.5B":    ATLAS_DIR / "Qwen2.5-1.5B_head_atlas.json",
    "Llama-3.2-1B": ATLAS_DIR / "Llama-3.2-1B_head_atlas.json",
}

REGIME_FILES = {
    "GPT-2":        REGIME_DIR / "regime_switching_gpt2-medium.json",
    "Qwen-0.5B":    REGIME_DIR / "regime_switching_Qwen_Qwen2.5-0.5B.json",
    "Qwen-1.5B":    REGIME_DIR / "regime_switching_Qwen_Qwen2.5-1.5B.json",
    "Llama-3.2-1B": REGIME_DIR / "regime_switching_unsloth_Llama-3.2-1B.json",
}

# Model HF IDs for weight loading
MODEL_IDS = {
    "GPT-2":        "gpt2-medium",
    "Qwen-0.5B":    "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B":    "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
}

# Architecture metadata
ARCH_META = {
    "GPT-2":        {"is_gqa": False, "kv_heads": None, "head_dim": 64},
    "Qwen-0.5B":    {"is_gqa": True,  "kv_heads": 8,   "head_dim": 64},
    "Qwen-1.5B":    {"is_gqa": True,  "kv_heads": 8,   "head_dim": 128},
    "Llama-3.2-1B": {"is_gqa": True,  "kv_heads": 8,   "head_dim": 64},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def effective_rank(s: np.ndarray) -> float:
    """
    Effective rank of a matrix given its singular values: exp(H(sigma)) 
    where H is the entropy of the normalised singular value distribution. 
    Roy & Vetterli (2007).
    """
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def top_k_sv(s: np.ndarray, k: int = 3) -> list:
    """Return the top-k singular values from an array of singular values."""
    padded = list(s[:k]) + [0.0] * max(0, k - len(s))
    return padded[:k]


def load_weight_features(model_key: str, n_layers: int, n_heads: int) -> dict:
    """
    Load model weights WITHOUT a forward pass and extract per-head structural
    features: norms, OV norm, QK/OV effective ranks, top-3 SVs.

    Returns dict keyed by (layer_idx, head_idx).
    """
    import gc
    import torch
    from transformers import AutoModelForCausalLM

    model_id = MODEL_IDS[model_key]
    print(f"  Loading weights for {model_id} (CPU, no grad)...")
    sys.stdout.flush()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
    model.eval()

    n_kv = getattr(model.config, "num_key_value_heads", n_heads)
    is_gqa = (n_kv != n_heads)
    head_dim = model.config.hidden_size // n_heads
    d_model = model.config.hidden_size

    features = {}
    for L in range(n_layers):
        # ---- resolve layer module ----
        if model_key == "GPT-2":
            attn = model.transformer.h[L].attn
            # GPT-2 uses a combined c_attn projection [Q, K, V] and c_proj
            c_attn_w = attn.c_attn.weight.detach().float().numpy()  # (d_model, 3*d_model)
            c_proj_w = attn.c_proj.weight.detach().float().numpy()  # (d_model, d_model)
            q_full = c_attn_w[:, :d_model].T          # (d_model, d_model) -> each col is a head
            k_full = c_attn_w[:, d_model:2*d_model].T
            v_full = c_attn_w[:, 2*d_model:].T
            o_full = c_proj_w.T
        else:
            # HuggingFace Qwen/Llama naming
            try:
                attn = model.model.layers[L].self_attn
            except AttributeError:
                attn = model.transformer.h[L].attn
            q_full = attn.q_proj.weight.detach().float().numpy()   # (n_heads*head_dim, d_model)
            k_full = attn.k_proj.weight.detach().float().numpy()   # (n_kv*head_dim, d_model)
            v_full = attn.v_proj.weight.detach().float().numpy()
            o_full = attn.o_proj.weight.detach().float().numpy()   # (d_model, n_heads*head_dim)

        for H in range(n_heads):
            # Slice per-head weight blocks
            if model_key == "GPT-2":
                wq = q_full[H*head_dim:(H+1)*head_dim, :]   # (head_dim, d_model)
                wk = k_full[H*head_dim:(H+1)*head_dim, :]
                wv = v_full[H*head_dim:(H+1)*head_dim, :]
                wo = o_full[:, H*head_dim:(H+1)*head_dim]   # (d_model, head_dim)
            else:
                wq = q_full[H*head_dim:(H+1)*head_dim, :]
                # GQA: K/V heads shared across groups of query heads
                kv_group = H // (n_heads // n_kv) if is_gqa else H
                wk = k_full[kv_group*head_dim:(kv_group+1)*head_dim, :]
                wv = v_full[kv_group*head_dim:(kv_group+1)*head_dim, :]
                wo = o_full[:, H*head_dim:(H+1)*head_dim]

            # Norms
            wq_norm = float(np.linalg.norm(wq, "fro"))
            wk_norm = float(np.linalg.norm(wk, "fro"))
            wv_norm = float(np.linalg.norm(wv, "fro"))
            wo_norm = float(np.linalg.norm(wo, "fro"))
            vq_ratio = wv_norm / (wq_norm + 1e-12)

            # OV product singular values: computing SVD of (d_model, d_model) is O(D^3) = very slow.
            # We use the QR trick for exact singular values in O(D H^2).
            # W_O is (d_model, head_dim). W_V is (head_dim, d_model).
            # W_O = Q1 R1. W_V^T = Q2 R2. 
            # Non-zero SVs of W_O W_V are exactly the SVs of (R1 @ R2.T) which is (head_dim, head_dim).
            _, r1 = np.linalg.qr(wo)     # (head_dim, head_dim)
            _, r2 = np.linalg.qr(wv.T)   # (head_dim, head_dim)
            core = r1 @ r2.T
            _, ov_s, _ = np.linalg.svd(core)
            
            # For Frobenius norm, ||W_O W_V||_F = sqrt(sum(SVs^2))
            ov_norm = float(np.sqrt(np.sum(ov_s**2)))
            ov_eff_rank = effective_rank(ov_s)
            ov_sv = top_k_sv(ov_s, k=3)

            # QK weight product: W_Q @ W_K^T → (head_dim, head_dim)
            qk_mat = wq @ wk.T
            _, qk_s, _ = np.linalg.svd(qk_mat)
            qk_eff_rank = effective_rank(qk_s)
            qk_sv = top_k_sv(qk_s, k=3)

            # GQA group features — architecture-agnostic derivations.
            # We do NOT use raw group ID (leaks arch identity via ID range).
            # Instead:
            #   gqa_group_size:        how many query heads share one KV head
            #   gqa_within_group_rank: this head's 0-indexed position inside its group
            # For MHA (GPT-2), both are 1 and 0 respectively.
            if is_gqa:
                group_size        = n_heads // n_kv
                within_group_rank = H % group_size
            else:
                group_size        = 1
                within_group_rank = 0

            features[(L, H)] = {
                "wq_norm": wq_norm,
                "wk_norm": wk_norm,
                "wv_norm": wv_norm,
                "wo_norm": wo_norm,
                "vq_ratio_weight": vq_ratio,
                "ov_norm": ov_norm,
                "ov_eff_rank": ov_eff_rank,
                "ov_sv1": ov_sv[0], "ov_sv2": ov_sv[1], "ov_sv3": ov_sv[2],
                "qk_eff_rank": qk_eff_rank,
                "qk_sv1": qk_sv[0], "qk_sv2": qk_sv[1], "qk_sv3": qk_sv[2],
                "head_dim": head_dim,
                "is_gqa": int(is_gqa),
                "gqa_group_size": group_size,
                "gqa_within_group_rank": within_group_rank,
            }

    del model
    gc.collect()
    return features


def load_locality_map(model_key: str) -> dict:
    """
    Parse the regime_switching JSON to extract a per-head locality score.
    The JSON only has top-10 stable/switchers, so we fill the rest with NaN.
    Returns dict (layer, head) -> locality score (mean across contexts).
    """
    path = REGIME_FILES[model_key]
    if not path.exists():
        warnings.warn(f"Regime switching file not found for {model_key}: {path}")
        return {}
    with open(path) as f:
        d = json.load(f)

    locality = {}
    for entry in d.get("top_10_most_stable", []):
        locality[(entry["layer"], entry["head"])] = \
            float(np.mean(list(entry["group_means"].values())))
    for entry in d.get("top_10_regime_switchers", []):
        locality[(entry["layer"], entry["head"])] = \
            float(np.mean(list(entry["group_means"].values())))
    return locality


def extract_atlas_features(head_record: dict) -> dict:
    """Parse the per-head dict from head_atlas.json into flat feature columns."""
    ep = head_record.get("entropy_profile", {})
    ag = head_record.get("attention_geometry", {})
    ss = head_record.get("softmax_saturation", {})
    return {
        "match_entropy":    ep.get("match_entropy", np.nan),
        "delta_collapse":   ep.get("delta_collapse", np.nan),
        "mean_distance":    ag.get("mean_distance", np.nan),
        "bos_mass":         ag.get("bos_mass", np.nan),
        "local_mass":       ag.get("local_mass", np.nan),
        "long_range_mass":  ag.get("long_range_mass", np.nan),
        "mean_max_attn":    ss.get("mean_max_attn", np.nan),
        "mean_entropy_ss":  ss.get("mean_entropy", np.nan),
        "vq_ratio_atlas":   head_record.get("vq_ratio", np.nan),
        "mean_output_norm": head_record.get("mean_output_norm", np.nan),
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_feature_bank(skip_weight_features: bool = False) -> pd.DataFrame:
    """
    Main loop: iterate over all 4 models' canonical labels, join atlas features,
    and optionally load weight features.

    Args:
        skip_weight_features: If True, skip model weight loading (faster for
                               syntax checking). Weight columns will be NaN.
    """
    with open(CANONICAL_LABELS_PATH) as f:
        canonical = json.load(f)

    records = []
    missing_atlas_count = 0

    for model_key, model_data in canonical["models"].items():
        atlas_path = ATLAS_FILES.get(model_key)
        if atlas_path is None or not atlas_path.exists():
            warnings.warn(f"Atlas file missing for {model_key}: {atlas_path}")
            atlas_heads = {}
        else:
            with open(atlas_path) as f:
                atlas_data = json.load(f)
            atlas_heads = atlas_data.get("heads", {})

        locality_map = load_locality_map(model_key)
        n_layers = model_data["n_layers"]
        heads_dict = model_data["heads"]

        # Optionally load weight features (big but still no GPU forward pass)
        if not skip_weight_features:
            n_heads_example = max(v["head_idx"] for v in heads_dict.values()) + 1
            weight_feats = load_weight_features(model_key, n_layers, n_heads_example)
        else:
            weight_feats = {}
            print(f"  [skip_weight_features=True] Skipping model weight loading for {model_key}")

        for head_key, head_info in heads_dict.items():
            L = head_info["layer"]
            H = head_info["head_idx"]
            label = head_info["label"].lower()  # 'local' | 'sink' | 'induction' | 'retrieval'
            rel_depth = head_info["relative_depth"]

            # Atlas features — this is the PRIMARY source of entropy/locality
            atlas_record = atlas_heads.get(f"{L}_{H}", None)
            if atlas_record is None:
                atlas_feats = {k: np.nan for k in [
                    "match_entropy", "delta_collapse", "mean_distance", "bos_mass",
                    "local_mass", "long_range_mass", "mean_max_attn", "mean_entropy_ss",
                    "vq_ratio_atlas", "mean_output_norm"
                ]}
                missing_atlas_count += 1
            else:
                atlas_feats = extract_atlas_features(atlas_record)

            # Locality from regime switching (partial coverage — NaN for unlogged heads)
            locality_score = locality_map.get((L, H), np.nan)

            # Weight features
            wf = weight_feats.get((L, H), {})

            row = {
                "model":        model_key,
                "layer":        L,
                "head":         H,
                "label":        label,
                "rel_depth":    rel_depth,
                "locality_score": locality_score,
                **atlas_feats,
                "wq_norm":      wf.get("wq_norm", np.nan),
                "wk_norm":      wf.get("wk_norm", np.nan),
                "wv_norm":      wf.get("wv_norm", np.nan),
                "wo_norm":      wf.get("wo_norm", np.nan),
                "vq_ratio_weight": wf.get("vq_ratio_weight", np.nan),
                "ov_norm":      wf.get("ov_norm", np.nan),
                "ov_eff_rank":  wf.get("ov_eff_rank", np.nan),
                "ov_sv1":       wf.get("ov_sv1", np.nan),
                "ov_sv2":       wf.get("ov_sv2", np.nan),
                "ov_sv3":       wf.get("ov_sv3", np.nan),
                "qk_eff_rank":  wf.get("qk_eff_rank", np.nan),
                "qk_sv1":       wf.get("qk_sv1", np.nan),
                "qk_sv2":       wf.get("qk_sv2", np.nan),
                "qk_sv3":       wf.get("qk_sv3", np.nan),
                "head_dim":     wf.get("head_dim", np.nan),
                "is_gqa":       wf.get("is_gqa", np.nan),
                "gqa_group_size":         wf.get("gqa_group_size", np.nan),
                "gqa_within_group_rank":  wf.get("gqa_within_group_rank", np.nan),
            }
            records.append(row)

    df = pd.DataFrame(records)
    total = len(df)
    missing_frac = missing_atlas_count / total

    print(f"\n[Feature Bank] {total} heads across {df['model'].nunique()} architectures")
    print(f"[Feature Bank] Missing atlas records: {missing_atlas_count} ({100*missing_frac:.1f}%)")

    # Fix 4: Hard abort/warn on missing atlas tolerance
    if missing_frac > MISSING_ATLAS_ABORT_THRESHOLD:
        raise RuntimeError(
            f"ABORT: {missing_atlas_count} heads ({100*missing_frac:.1f}%) are missing from "
            f"the atlas JSONs, exceeding the {100*MISSING_ATLAS_ABORT_THRESHOLD:.0f}% tolerance. "
            "Behavioral features (entropy, distance, bos_mass) for these heads will be NaN, "
            "creating measurement inconsistency with the labels (which used the same probe). "
            "Reconcile the atlas files before running Phase 0."
        )
    elif missing_frac > MISSING_ATLAS_WARN_THRESHOLD:
        warnings.warn(
            f"[WARNING] {missing_atlas_count} heads ({100*missing_frac:.1f}%) missing from atlas. "
            "Proceeding, but note their behavioral features are NaN and will be median-imputed "
            "by 01_phase0_gate.py. This introduces minor measurement inconsistency."
        )

    print(f"[Feature Bank] Class distribution:\n{df['label'].value_counts()}")
    print(f"[Feature Bank] NaN summary:\n{df.isna().sum()}")

    out_path = OUTPUT_DIR / "feature_bank.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Feature Bank] Saved to {out_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip model weight loading (for fast syntax check)")
    args = parser.parse_args()
    extract_feature_bank(skip_weight_features=args.skip_weights)
