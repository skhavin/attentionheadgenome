import torch

def fix_sink(attn_weights, V, num_sink_tokens=4):
    B, H, N, d = V.shape
    sink_V = V[:, :, :num_sink_tokens, :]
    sink_w = attn_weights[:, :, :, :num_sink_tokens].mean(dim=2, keepdim=True)
    
    # NORMALIZE
    sink_w = sink_w / sink_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    
    out = torch.einsum("bhnk,bhkd->bhnd", sink_w, sink_V)
    out = out.expand(B, H, N, d)
    return out

def fix_local(V, window_size=64):
    import torch.nn.functional as F
    B, H, N, d = V.shape
    v_t = V.permute(0, 1, 3, 2).contiguous()   # [B, H, d, N]
    v_in = v_t.view(B * H * d, 1, N)

    kernel = torch.ones(1, 1, window_size, dtype=V.dtype, device=V.device) / window_size
    v_padded = F.pad(v_in, (window_size - 1, 0))
    
    out_flat = F.conv1d(v_padded, kernel, padding=0)
    
    counts = torch.arange(1, N + 1, dtype=V.dtype, device=V.device)
    counts = torch.clamp(counts, max=window_size)
    out_flat = out_flat * (window_size / counts.view(1, 1, N))

    out = out_flat.view(B, H, d, N).permute(0, 1, 3, 2)
    return out

# I'll just rewrite substitutes.py directly and re-run audit_heads on a small sample.
