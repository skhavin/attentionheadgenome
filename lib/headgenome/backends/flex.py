"""
headgenome.backends.flex
────────────────────────
FlexAttention backend using PyTorch's mask_mod + create_block_mask.
Skips blocks entirely (true FLOP reduction) rather than applying -inf masks.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    _flex_available = True
except ImportError:
    _flex_available = False


def build_headgenome_flex_fn(
    role_map: Dict[Tuple[int, int], str],  # {(layer, head): role}
    num_heads: int,
    window: int,
    sink_size: int,
    preserve_roles: Set[str],
    num_layers: int,
):
    """
    Returns a per-layer callable: flex_forward(layer_idx, q, k, v, scale) → out
    Each layer gets its own BlockMask computed from its head roles.
    """
    if not _flex_available:
        raise RuntimeError("FlexAttention is not available. Requires PyTorch >= 2.5")

    def make_mask_mod(layer_roles: Dict[int, str]):
        """Create a mask_mod function for a specific layer's head role assignments."""
        # Precompute which heads are critical (full attention)
        critical_set = frozenset(h for h, role in layer_roles.items() if role in preserve_roles)
        sink_set     = frozenset(h for h, role in layer_roles.items() if role == "sink")
        W = window
        S = sink_size

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            is_sink_tok = kv_idx < S
            is_local = (q_idx - kv_idx) <= W

            # Use torch.where chains — mask_mod must be torch-compilable
            is_critical = torch.zeros_like(causal)
            for ch in critical_set:
                is_critical = is_critical | (h == ch)

            is_sink_head = torch.zeros_like(causal)
            for sh in sink_set:
                is_sink_head = is_sink_head | (h == sh)

            # Critical: full causal
            full_mask  = causal
            # Local / sink heads: causal & (sink_tok | local_window)
            local_mask = causal & (is_sink_tok | is_local)

            return torch.where(is_critical, full_mask, local_mask)

        return mask_mod

    # Pre-build one BlockMask generator per layer
    layer_mask_mods = {}
    for li in range(num_layers):
        layer_roles = {hi: role_map.get((li, hi), "local") for hi in range(num_heads)}
        layer_mask_mods[li] = make_mask_mod(layer_roles)

    def flex_forward(layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        """
        Run FlexAttention for one layer.
        q, k, v: (B, H, S, D)
        """
        B, H, S, D = q.shape
        mask_mod = layer_mask_mods[layer_idx]

        block_mask = create_block_mask(
            mask_mod,
            B=None, H=None, Q_LEN=S, KV_LEN=S,
            device=q.device,
            _compile=False,
        )
        out = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
        return out

    return flex_forward
