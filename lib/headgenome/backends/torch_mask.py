"""
headgenome.backends.torch_mask
──────────────────────────────
Correctness reference backend: additive -inf mask via forward pre-hooks.
No FLOP savings (still runs full attention), but exact numerical parity
with the FlexAttention backend — use this to validate correctness first.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

import torch
from transformers import PreTrainedModel


def _get_attn_module(model, layer_idx):
    for path in ["model.layers", "transformer.h", "model.decoder.layers"]:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            layer = obj[layer_idx]
            for attr in ["self_attn", "attn", "attention", "self_attention"]:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
        except Exception:
            continue
    raise AttributeError(f"Cannot find attention module for layer {layer_idx}")


class TorchMaskHooks:
    """
    Installs forward pre-hooks on every attention layer.
    The hook injects an additive mask into kwargs["attention_mask"].

    This is the correctness reference — not the speedup path.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        role_map: Dict[Tuple[int, int], str],
        num_layers: int,
        num_heads: int,
        window: int,
        sink_size: int,
        preserve_roles: Set[str],
    ):
        self.handles: List = []
        self.num_heads = num_heads

        for li in range(num_layers):
            attn_module = _get_attn_module(model, li)
            layer_roles = {hi: role_map.get((li, hi), "local") for hi in range(num_heads)}
            handle = attn_module.register_forward_pre_hook(
                self._make_hook(layer_roles, window, sink_size, preserve_roles),
                with_kwargs=True,
            )
            self.handles.append(handle)

    @staticmethod
    def _make_hook(layer_roles, W, S, preserve_roles):
        def pre_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return args, kwargs

            q_len  = hidden_states.shape[1]
            device = hidden_states.device
            dtype  = hidden_states.dtype

            cache_pos = kwargs.get("cache_position")
            if cache_pos is not None:
                q_pos  = cache_pos.unsqueeze(1)
                kv_len = int(cache_pos[-1].item()) + 1
                k_pos  = torch.arange(kv_len, device=device).unsqueeze(0)
            else:
                q_pos  = torch.arange(q_len, device=device).unsqueeze(1)
                kv_len = q_len
                k_pos  = torch.arange(kv_len, device=device).unsqueeze(0)

            causal_ok   = q_pos >= k_pos
            is_sink_tok = k_pos < S
            is_local    = (q_pos - k_pos) < W

            zero = torch.tensor(0.0, dtype=dtype, device=device)
            neginf = torch.tensor(float("-inf"), dtype=dtype, device=device)

            full_mask  = torch.where(causal_ok, zero, neginf)
            local_mask = torch.where(causal_ok & (is_sink_tok | is_local), zero, neginf)

            H = len(layer_roles)
            role_mask = torch.empty(1, H, q_len, kv_len, dtype=dtype, device=device)
            for hi in range(H):
                role = layer_roles.get(hi, "local")
                if role in preserve_roles:
                    role_mask[0, hi] = full_mask
                else:
                    role_mask[0, hi] = local_mask

            existing = kwargs.get("attention_mask")
            if existing is not None:
                try:
                    new_mask = existing + role_mask
                except Exception:
                    new_mask = role_mask
            else:
                new_mask = role_mask

            kwargs = dict(kwargs, attention_mask=new_mask)
            return args, kwargs

        return pre_hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def apply_torch_mask(
    model: PreTrainedModel,
    role_map: Dict[Tuple[int, int], str],
    num_layers: int,
    num_heads: int,
    window: int,
    sink_size: int,
    preserve_roles: Set[str],
) -> TorchMaskHooks:
    """Install hooks and return handle object. Call .remove() to undo."""
    return TorchMaskHooks(
        model=model,
        role_map=role_map,
        num_layers=num_layers,
        num_heads=num_heads,
        window=window,
        sink_size=sink_size,
        preserve_roles=preserve_roles,
    )
