"""
01_triage_and_schema.py

This file is the single source of truth for which features are allowed
into the genome dataset and which are excluded to prevent circularity.

It also defines the reusable extraction functions for static weight geometry
(norms, SVD decomposition, effective rank) that all downstream scripts import.

NO model loading happens here. Pure definitions only.
"""

import numpy as np
import torch
from scipy import stats

# =============================================================================
# FEATURE BINS
# =============================================================================

# BIN 1: Causally upstream of function.
# These are derived purely from static weight matrices (W_Q, W_K, W_V, W_O).
# They predate any forward pass, so they cannot be circular w.r.t. any label
# derived from activation or attention patterns.
BIN_1_FEATURES = [
    # Raw norms
    "W_Q_norm", "W_K_norm", "W_V_norm", "W_O_norm",
    "W_Q_frobenius_norm", "W_K_frobenius_norm", "W_V_frobenius_norm", "W_O_frobenius_norm",
    "W_Q_spectral_norm", "W_K_spectral_norm", "W_V_spectral_norm", "W_O_spectral_norm",
    # Cross-weight ratios (the core features from our zero-shot formula)
    "V_to_Q_norm_ratio", "Q_to_K_norm_ratio", "V_to_K_norm_ratio", "O_to_V_norm_ratio",
    # Weight statistics
    "W_Q_mean", "W_K_mean", "W_V_mean", "W_O_mean",
    "W_Q_std", "W_K_std", "W_V_std", "W_O_std",
    "W_Q_sparsity", "W_K_sparsity", "W_V_sparsity", "W_O_sparsity",
    # QK matrix geometry (W_Q @ W_K.T)
    "QK_frobenius_norm", "QK_spectral_norm", "QK_effective_rank", "QK_condition_number",
    "QK_top_1_sv", "QK_top_2_sv", "QK_top_3_sv",
    "QK_singular_value_entropy", "QK_anisotropy",
    # OV matrix geometry (W_V @ W_O.T)
    "OV_frobenius_norm", "OV_spectral_norm", "OV_effective_rank", "OV_condition_number",
    "OV_top_1_sv", "OV_top_2_sv", "OV_top_3_sv",
    "OV_singular_value_entropy", "OV_anisotropy",
    # Architecture metadata
    "layer_index", "head_index", "relative_depth",
    "num_layers", "num_attention_heads", "num_kv_heads", "head_dim",
]

# BIN 2: Behavioral features computed independently of the label procedure.
# These require a forward pass but are computed on a HELD-OUT prompt set
# that was never used to derive the canonical Phase 1 entropy-collapse labels.
# They measure runtime behavior, not the labeling criterion itself.
BIN_2_FEATURES = [
    # Pre-softmax score stats
    "qk_score_mean", "qk_score_std", "qk_score_max", "qk_score_min",
    "qk_top1_minus_top2", "qk_top1_zscore",
    "qk_score_gini", "qk_score_kurtosis",
    # Phase 2 noise floor diagnostics (key for understanding early-exit failure)
    "qk_score_noise_floor", "qk_false_spike_rate", "first_hit_distance",
    "true_target_rank_by_qk_score", "true_target_score_percentile",
    # Positional distance behavior
    "mean_attention_distance", "median_attention_distance",
    "local_mass_32", "local_mass_64", "local_mass_128", "local_mass_256",
    "long_range_mass_512", "long_range_mass_1024",
    "distance_decay_slope",
    # Sink behavior
    "bos_mass", "first_4_token_mass",
    "punctuation_mass", "delimiter_mass",
    # Token-type attention mass
    "proper_noun_mass", "number_mass", "rare_token_mass",
    "function_word_mass", "content_word_mass",
    # Softmax distribution shape
    "attention_entropy_mean", "attention_gini",
    "attention_top1_mass", "attention_top5_mass",
]

# BIN 3: EXCLUDED. Label restatements.
# These are derived from the same probe family (NIAH, copy/induction tasks,
# entropy collapse) that generated the canonical labels in Table 1.
# Including them would be tautological — we'd be predicting labels using
# a re-description of the labeling criterion itself.
# This list is enforced by the assertion below.
BIN_3_FEATURES = [
    "retrieval_head_precision", "retrieval_head_recall",
    "needle_retrieval_success", "needle_attention_mass",
    "induction_score", "copy_score", "AB_A_to_B_score",
    "attention_entropy_delta",  # This IS the Phase 1 label criterion
    "sink_dependency_for_niah", "sink_dependency_for_ppl",
]

# All approved features for downstream scripts to import
APPROVED_FEATURES = BIN_1_FEATURES + BIN_2_FEATURES


def assert_no_bin3_leak(df):
    """
    Hard assertion: Bin-3 features must never appear in the final dataframe.
    This is the same class of bug that broke Phase 0.
    Call this immediately after creating or loading any dataframe.
    """
    leaked = set(BIN_3_FEATURES).intersection(set(df.columns))
    assert len(leaked) == 0, (
        f"CIRCULARITY BUG: Bin-3 label-restatement features leaked into dataset! "
        f"Offending columns: {leaked}"
    )


# =============================================================================
# STATIC EXTRACTION FUNCTIONS (Bin 1 — weight geometry)
# =============================================================================

def safe_svd(matrix, k=3):
    """
    Compute singular value decomposition of a matrix.
    Returns (singular_values, effective_rank, condition_number, anisotropy, sv_entropy).
    Handles numerical edge cases gracefully.
    """
    try:
        # Use torch SVD for speed
        _, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
        s = s.cpu().numpy()

        # Effective rank: exp(entropy of normalized squared singular values)
        s2 = s ** 2
        s2_norm = s2 / (s2.sum() + 1e-10)
        sv_entropy = float(-np.sum(s2_norm * np.log(s2_norm + 1e-10)))
        effective_rank = float(np.exp(sv_entropy))

        # Condition number: ratio of largest to smallest non-zero singular value
        nonzero_s = s[s > 1e-6]
        condition_number = float(nonzero_s[0] / nonzero_s[-1]) if len(nonzero_s) > 1 else 0.0

        # Anisotropy: how dominated by the top singular value (1 = totally dominated)
        anisotropy = float(s[0] ** 2 / (s2.sum() + 1e-10))

        top_k_svs = [float(s[i]) if i < len(s) else 0.0 for i in range(k)]

        return top_k_svs, effective_rank, condition_number, anisotropy, sv_entropy

    except Exception as e:
        print(f"  [WARN] SVD failed: {e}")
        return [0.0] * k, 0.0, 0.0, 0.0, 0.0


def extract_bin1_features(W_Q, W_K, W_V, W_O, layer_idx, head_idx, n_layers, n_heads, n_kv_heads):
    """
    Extract all Bin-1 static weight features for a single head.

    Args:
        W_Q: [head_dim, hidden_dim] - Query projection for this head
        W_K: [head_dim, hidden_dim] - Key projection for this head (may be shared in GQA)
        W_V: [head_dim, hidden_dim] - Value projection for this head (may be shared in GQA)
        W_O: [hidden_dim, head_dim] - Output projection for this head

    Returns:
        dict of feature_name -> float
    """
    feats = {}

    # Identity
    feats["layer_index"] = layer_idx
    feats["head_index"] = head_idx
    feats["relative_depth"] = layer_idx / max(n_layers - 1, 1)
    feats["num_layers"] = n_layers
    feats["num_attention_heads"] = n_heads
    feats["num_kv_heads"] = n_kv_heads
    feats["head_dim"] = W_Q.shape[0]

    # Individual weight norms
    for name, W in [("Q", W_Q), ("K", W_K), ("V", W_V), ("O", W_O)]:
        w = W.float()
        feats[f"W_{name}_norm"] = float(torch.norm(w))
        feats[f"W_{name}_frobenius_norm"] = float(torch.norm(w, p="fro"))
        feats[f"W_{name}_mean"] = float(w.mean())
        feats[f"W_{name}_std"] = float(w.std())
        feats[f"W_{name}_sparsity"] = float((w.abs() < 1e-5).float().mean())
        # Spectral norm (largest singular value)
        _, s, _ = torch.linalg.svd(w, full_matrices=False)
        feats[f"W_{name}_spectral_norm"] = float(s[0])

    # Cross-weight ratios (our core zero-shot formula features)
    q_norm = feats["W_Q_norm"] + 1e-10
    k_norm = feats["W_K_norm"] + 1e-10
    v_norm = feats["W_V_norm"] + 1e-10
    o_norm = feats["W_O_norm"] + 1e-10

    feats["V_to_Q_norm_ratio"] = v_norm / q_norm
    feats["Q_to_K_norm_ratio"] = q_norm / k_norm
    feats["V_to_K_norm_ratio"] = v_norm / k_norm
    feats["O_to_V_norm_ratio"] = o_norm / v_norm

    # QK circuit geometry: W_Q @ W_K.T
    QK = W_Q.float() @ W_K.float().T
    top_svs, eff_rank, cond, aniso, sv_ent = safe_svd(QK, k=3)
    feats["QK_frobenius_norm"] = float(torch.norm(QK, p="fro"))
    feats["QK_spectral_norm"] = top_svs[0]
    feats["QK_top_1_sv"] = top_svs[0]
    feats["QK_top_2_sv"] = top_svs[1]
    feats["QK_top_3_sv"] = top_svs[2]
    feats["QK_effective_rank"] = eff_rank
    feats["QK_condition_number"] = cond
    feats["QK_anisotropy"] = aniso
    feats["QK_singular_value_entropy"] = sv_ent

    # OV circuit geometry: W_V @ W_O.T
    OV = W_V.float() @ W_O.float().T
    top_svs, eff_rank, cond, aniso, sv_ent = safe_svd(OV, k=3)
    feats["OV_frobenius_norm"] = float(torch.norm(OV, p="fro"))
    feats["OV_spectral_norm"] = top_svs[0]
    feats["OV_top_1_sv"] = top_svs[0]
    feats["OV_top_2_sv"] = top_svs[1]
    feats["OV_top_3_sv"] = top_svs[2]
    feats["OV_effective_rank"] = eff_rank
    feats["OV_condition_number"] = cond
    feats["OV_anisotropy"] = aniso
    feats["OV_singular_value_entropy"] = sv_ent

    return feats


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def compute_gini(arr):
    """Gini coefficient of an array — measures inequality/concentration."""
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def compute_entropy(probs, eps=1e-10):
    """Shannon entropy of a probability distribution."""
    probs = np.array(probs, dtype=float)
    probs = probs / (probs.sum() + eps)
    return float(-np.sum(probs * np.log(probs + eps)))


if __name__ == "__main__":
    print("01_triage_and_schema.py loaded.")
    print(f"  Bin 1 features: {len(BIN_1_FEATURES)}")
    print(f"  Bin 2 features: {len(BIN_2_FEATURES)}")
    print(f"  Bin 3 excluded: {len(BIN_3_FEATURES)}")
    print(f"  Total approved: {len(APPROVED_FEATURES)}")

    # Sanity check: no BIN_3 feature should accidentally be in APPROVED_FEATURES
    overlap = set(BIN_3_FEATURES).intersection(set(APPROVED_FEATURES))
    assert len(overlap) == 0, f"Schema bug: BIN_3 overlaps APPROVED: {overlap}"
    print("  [OK] No Bin-3 features in approved list.")
