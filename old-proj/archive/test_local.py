import torch
import torch.nn.functional as F

def local_substitute(V, window_size=64):
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

def exact_local(V, window_size=64):
    B, H, N, d = V.shape
    out = torch.zeros_like(V)
    for i in range(N):
        start = max(0, i - window_size + 1)
        out[:, :, i, :] = V[:, :, start:i+1, :].mean(dim=2)
    return out

V = torch.randn(1, 1, 100, 4)
sub = local_substitute(V, 64)
exact = exact_local(V, 64)

diff = (sub - exact).abs().max().item()
print("Local substitute error:", diff)
