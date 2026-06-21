# Wrap HuggingFace's past_key_values to apply a retention mask.
# Drops KV entries for tokens not in the mask.

import torch


def apply_retention_mask(past_key_values, masks, budget, device="cuda"):
    """Prune KV cache to keep only the token positions marked True in the mask.
    All heads share the same mask (HF requires uniform seq_len across heads)."""
    if past_key_values is None:
        return None

    seq_len = past_key_values[0][0].shape[2]

    # All masks are identical (global mask), just grab the first one
    first_mask = next(iter(masks.values()))
    mask_len = min(len(first_mask), seq_len)

    # Build index tensor of positions to keep
    mask_tensor = torch.zeros(seq_len, dtype=torch.bool)
    mask_tensor[:mask_len] = torch.tensor(first_mask[:mask_len], dtype=torch.bool)

    indices = mask_tensor.nonzero(as_tuple=True)[0]

    # Enforce budget limit
    if len(indices) > budget:
        indices = indices[:budget]

    indices = indices.to(device)

    # Slice every layer's K and V tensors
    pruned = []
    for layer_kv in past_key_values:
        k, v = layer_kv  # each is (batch, heads, seq_len, head_dim)
        pruned.append((k.index_select(2, indices), v.index_select(2, indices)))

    return tuple(pruned)
