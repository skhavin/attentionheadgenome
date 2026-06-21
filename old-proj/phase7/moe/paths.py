import torch
import torch.nn.functional as F
import math

def sink_path(v: torch.Tensor, num_sink_tokens: int = 4) -> torch.Tensor:
    """
    O(1) approximation: Uniform average of the first `num_sink_tokens`.
    Input:
        v: [batch, num_heads, seq_len, head_dim]
    Output:
        out: same shape as v
    """
    seq_len = v.size(-2)
    # If seq_len is smaller than num_sink_tokens, take what we have
    actual_sink = min(seq_len, num_sink_tokens)
    sink_v = v[..., :actual_sink, :]
    
    # Take mean over seq_len dimension
    mean_v = sink_v.mean(dim=-2, keepdim=True)
    # Expand to match sequence length
    return mean_v.expand(*v.shape[:-2], seq_len, v.size(-1))

def local_path(v: torch.Tensor, window_size: int = 64) -> torch.Tensor:
    """
    O(N*W) approximation: Causal uniform 1D convolution over the last `window_size` tokens.
    Input:
        v: [batch, num_heads, seq_len, head_dim]
    Output:
        out: same shape as v
    """
    shape = v.shape
    v = v.reshape(-1, shape[-2], shape[-1]) # [B*H, N, d]
    B_H, N, d = v.shape

    # Permute for conv1d: [B*H*d, 1, N]
    v_t = v.transpose(1, 2)
    v_in = v_t.reshape(B_H * d, 1, N)
    
    kernel = torch.ones(1, 1, window_size, dtype=v.dtype, device=v.device) / window_size
    
    # Pad past tokens for causal effect
    v_padded = F.pad(v_in, (window_size - 1, 0))
    out_flat = F.conv1d(v_padded, kernel, padding=0) # [B*H*d, 1, N]
    
    # Correct scale for initial tokens that have less than window_size history
    counts = torch.arange(1, N + 1, dtype=v.dtype, device=v.device)
    counts = torch.clamp(counts, max=window_size)
    out_flat = out_flat * (window_size / counts.view(1, 1, N))
    
    out = out_flat.view(B_H, d, N).transpose(1, 2)
    return out.view(shape)

def full_path(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    O(N^2) exact computation: Full causal softmax attention.
    Inputs:
        q, k, v: [batch, num_heads, seq_len, head_dim]
    Output:
        out: same shape
    """
    head_dim = q.size(-1)
    seq_len = q.size(-2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
    scores.masked_fill_(causal_mask, float('-inf'))
    
    probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out

def recurrence_path(v: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    """
    O(N) Exponential Moving Average (vectorized using F.conv1d for speed).
    Input:
        v: [batch, num_heads, seq_len, head_dim]
    Output:
        out: same shape
    """
    shape = v.shape
    v = v.reshape(-1, shape[-2], shape[-1]) # [B*H, N, d]
    B_H, N, d = v.shape
    
    # Truncate kernel for performance, alpha^256 is effectively zero
    K = min(N, 256)
    
    device, dtype = v.device, v.dtype
    powers = torch.arange(K, device=device, dtype=torch.float32)
    kernel = (1 - alpha) * (alpha ** powers)
    kernel = kernel.to(dtype).view(1, 1, K)
    
    v_t = v.transpose(1, 2).reshape(B_H * d, 1, N)
    v_padded = F.pad(v_t, (K - 1, 0))
    out_flat = F.conv1d(v_padded, kernel)
    out = out_flat.view(B_H, d, N).transpose(1, 2)
    
    # Fix initial token scale (v[0] weight should be alpha^t, not (1-alpha)*alpha^t)
    correction_weights = (alpha ** torch.arange(N, device=device, dtype=torch.float32)).to(dtype).view(1, N, 1)
    correction = v[:, 0:1, :] * (correction_weights * alpha)
    out = out + correction
    out[:, 0, :] = v[:, 0, :]
    
    return out.view(shape)
