# Convert a predicted prototype into a binary retention mask.
# The mask says which token positions to KEEP in the KV cache.
# The budget is the TOTAL number of tokens to keep (not per-head).

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
from config import PHASE2_DIR, PROTOTYPES_PATH, KV_BUDGETS


def score_tokens_by_prototypes(prototypes, predictions, seq_len, budget=256):
    """Score each token position by how important it is across all heads.
    Returns a single (seq_len,) score array — higher = more important to keep."""

    scores = np.zeros(seq_len, dtype=np.float64)

    for (layer, head) in sorted(prototypes.keys()):
        pred_cluster = predictions.get((layer, head), 0)
        centroid = prototypes[(layer, head)]["centroids"][pred_cluster]

        # The centroid is a histogram: centroid[d] = how much this head
        # attends to a token d positions back from the query.
        #
        # For token at position p:
        #   - It can be attended by query at position q where q > p
        #   - The attention weight from q to p is centroid[q - p]
        #   - Token p's importance = sum over all q > p of centroid[q - p]
        #
        # This is equivalent to a convolution: for each distance d,
        # all tokens that are d positions before some query get centroid[d] credit.
        # Token p gets credit from queries at p+1, p+2, ..., p+max_d.
        # Number of such queries = min(seq_len - p - 1, max_d).

        max_d = min(len(centroid), seq_len)
        # For each token position p, compute sum of centroid[:min(max_d, seq_len-p)]
        # This means early tokens get attended by MORE queries → higher score.
        # But we want a smooth score that differentiates positions.
        cumsum = np.cumsum(centroid[:max_d])

        for p in range(seq_len):
            # How far can queries reach back to this token?
            reach = min(max_d, seq_len - p)
            if reach > 0:
                scores[p] += cumsum[reach - 1]

    # Add position-based tiebreakers to ensure monotonic selection
    # Attention sinks: token 0 is critical
    scores[0] += scores.max() * 10.0

    # Recency: last few tokens matter for continuation, proportional to budget
    recency_bonus = np.zeros(seq_len)
    recency_window = min(max(8, budget // 16), seq_len)
    for i in range(recency_window):
        recency_bonus[seq_len - 1 - i] = scores.max() * max(0.5, 5.0 - i * 0.5)
    scores += recency_bonus

    # Small position-based tiebreaker so argsort is deterministic
    scores += np.linspace(0, 0.001, seq_len)

    return scores


def predict_retention_mask(prototypes, predictions, seq_len, budget):
    """Build a single global retention mask: which token positions to keep.
    Returns dict of (layer, head) -> boolean mask (all heads share the same mask
    because HuggingFace requires uniform seq_len across heads in a layer)."""

    scores = score_tokens_by_prototypes(prototypes, predictions, seq_len, budget)

    # Keep top-budget tokens
    actual_budget = min(budget, seq_len)
    top_indices = np.argsort(scores)[-actual_budget:]
    mask = np.zeros(seq_len, dtype=bool)
    mask[top_indices] = True

    # Return same mask for all (layer, head) pairs
    masks = {}
    for key in prototypes.keys():
        masks[key] = mask

    return masks


def main():
    # Demo: build masks and show how many tokens are kept at each budget
    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)

    pred_path = os.path.join(PHASE2_DIR, "predictions.pkl")
    with open(pred_path, "rb") as f:
        all_predictions = pickle.load(f)

    predictions = all_predictions[0]

    for budget in KV_BUDGETS:
        masks = predict_retention_mask(prototypes, predictions, 512, budget)
        first_key = sorted(masks.keys())[0]
        kept = masks[first_key].sum()
        kept_indices = np.where(masks[first_key])[0]
        print(f"  Budget {budget:4d}: keeping {kept}/{512} tokens, first few: {kept_indices[:8]}")

if __name__ == "__main__":
    main()
