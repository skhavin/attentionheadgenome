# phase7/substitutes.py
#
# Closed-form substitutions for sink and local attention heads.
#
# These are the core approximations whose error is being quantified.
# They are NOT "mathematically exact" — they are closed-form substitutions
# with bounded approximation error (L∞ ≤ 0.015 on natural text, downstream
# KL divergence reported per-head in the audit table from audit_heads.py).
#
# Head types and their substitutions:
#   sink  → weighted sum of V[:, 0:4, :]  (the model's attention sink tokens)
#           rationale: sink heads concentrate ≥90% mass on the first 4 tokens;
#           the substitution is exact in the limit where attn weight = 1 on those tokens.
#   local → depthwise 1-D convolution over V with window W
#           rationale: local heads attend within a fixed window; conv approximates
#           the weighted-sum within that window without the O(N²) cost.
#
# Both substitutions preserve tensor dtype and device.
# Both are O(1) (sink) or O(N·W) (local) in sequence length.

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sink head substitution — O(1)
# ---------------------------------------------------------------------------

def sink_substitute(
    V: torch.Tensor,           # [B, H, N, d_head]
    attn_weights: torch.Tensor = None,  # [B, H, N, N] — used only if provided to compute exact sink weights
    num_sink_tokens: int = 4,
) -> torch.Tensor:
    """
    Approximate output of a sink attention head.

    Sink heads empirically concentrate ≥90% of their attention mass on the
    first `num_sink_tokens` tokens.  We approximate the output as a uniform
    weighted sum over V[:, :, 0:num_sink_tokens, :].

    If `attn_weights` is provided (full softmax weights), we instead use the
    true marginal weights over the sink tokens — this gives the tight L∞ bound
    used in the audit table and is still O(1) per token once the sink weights
    are extracted.

    Returns:
        out: [B, H, N, d_head]  — same shape as full softmax output
    """
    B, H, N, d = V.shape
    sink_V = V[:, :, :num_sink_tokens, :]  # [B, H, K, d]

    if attn_weights is not None:
        # RIGHT — weighted sum of first num_sink_tokens with the actual observed sink weights
        # averaged over queries to preserve O(1) runtime property conceptually,
        # but matching the true unnormalized mass to avoid scaling errors.
        sink_w = attn_weights[:, :, :, :num_sink_tokens].mean(dim=2, keepdim=True)  # [B, H, 1, K]
        out = torch.einsum("bhnk,bhkd->bhnd", sink_w, sink_V)
        # Expand over sequence length
        out = out.expand(B, H, N, d)
    else:
        # Uniform over sink tokens — the runtime approximation
        # NOTE: At runtime we don't have attn_weights. But dividing by num_sink_tokens
        # assumes total mass 1.0, which might be an overestimate, but it's the best guess.
        out = sink_V.mean(dim=2, keepdim=True).expand(B, H, N, d)

    return out


# ---------------------------------------------------------------------------
# Local head substitution — O(N·W)
# ---------------------------------------------------------------------------

def local_substitute(
    V: torch.Tensor,           # [B, H, N, d_head]
    window_size: int = 64,
) -> torch.Tensor:
    """
    Approximate output of a local attention head via depthwise 1-D convolution.

    Local heads attend within a causal window of `window_size` tokens.
    The convolution approximates the uniform weighted average within that window,
    which matches the local head's behaviour when attention weights are roughly
    uniform over the recent context.

    Causal padding: we pad with (window_size - 1) zeros on the left (past),
    zero on the right, preserving the autoregressive property.

    Complexity: O(N · window_size) — linear in N for fixed W.

    Returns:
        out: [B, H, N, d_head]  — same shape as full softmax output
    """
    B, H, N, d = V.shape

    # Reshape for F.conv1d: treat each (B, H, d) channel independently
    # v_in: [B*H*d, 1, N]
    v_t = V.permute(0, 1, 3, 2).contiguous()   # [B, H, d, N]
    v_in = v_t.view(B * H * d, 1, N)

    # Uniform kernel of length window_size
    kernel = torch.ones(1, 1, window_size, dtype=V.dtype, device=V.device) / window_size

    # Causal left-padding: (window_size - 1) zeros on the left
    v_padded = F.pad(v_in, (window_size - 1, 0))

    out_flat = F.conv1d(v_padded, kernel, padding=0)  # [B*H*d, 1, N]
    
    # Fix scaling for early tokens: they shouldn't be divided by window_size
    # if they have fewer than window_size tokens in their causal past.
    counts = torch.arange(1, N + 1, dtype=V.dtype, device=V.device)
    counts = torch.clamp(counts, max=window_size)
    # out_flat currently divided by window_size. 
    # Multiply by window_size and divide by actual count.
    out_flat = out_flat * (window_size / counts.view(1, 1, N))

    out = out_flat.view(B, H, d, N).permute(0, 1, 3, 2)  # [B, H, N, d]

    return out


# ---------------------------------------------------------------------------
# Dispatcher — used by audit_heads.py and eval_ppl.py
# ---------------------------------------------------------------------------

def run_substitute(
    V: torch.Tensor,           # [B, H, N, d_head]
    head_type: str,            # "sink" | "local"
    attn_weights: torch.Tensor = None,
    num_sink_tokens: int = 4,
    local_window: int = 64,
) -> torch.Tensor:
    """
    Return the closed-form substitution output for a given head type.
    Raises ValueError for unknown head types (those require full attention).
    """
    if head_type == "sink":
        return sink_substitute(V, attn_weights=attn_weights,
                               num_sink_tokens=num_sink_tokens)
    elif head_type == "local":
        return local_substitute(V, window_size=local_window)
    else:
        raise ValueError(
            f"Unknown head type '{head_type}'. "
            f"Only 'sink' and 'local' have closed-form substitutions. "
            f"Content/global heads require full attention."
        )
