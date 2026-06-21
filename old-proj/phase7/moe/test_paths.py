import torch
from paths import sink_path, local_path, full_path

def test_paths():
    B, H, N, D = 2, 4, 128, 64
    
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    v = torch.randn(B, H, N, D, requires_grad=True)
    
    # 1. Test full path
    out_full = full_path(q, k, v)
    assert out_full.shape == (B, H, N, D), f"Full path shape mismatch: {out_full.shape}"
    
    out_full.sum().backward()
    assert q.grad is not None and torch.abs(q.grad).sum() > 0
    assert k.grad is not None and torch.abs(k.grad).sum() > 0
    assert v.grad is not None and torch.abs(v.grad).sum() > 0
    
    # Zero grads
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    
    # 2. Test sink path
    out_sink = sink_path(v, num_sink_tokens=4)
    assert out_sink.shape == (B, H, N, D), f"Sink path shape mismatch: {out_sink.shape}"
    
    # First 4 tokens should be uniform average
    v_sink_avg = v[:, :, :4, :].mean(dim=-2, keepdim=True)
    assert torch.allclose(out_sink[:, :, 0, :], v_sink_avg[:, :, 0, :])
    assert torch.allclose(out_sink[:, :, N-1, :], v_sink_avg[:, :, 0, :])
    
    out_sink.sum().backward()
    # Only v should have gradients, and only in the first 4 tokens
    assert q.grad is None or torch.abs(q.grad).sum() == 0
    assert k.grad is None or torch.abs(k.grad).sum() == 0
    assert v.grad is not None
    assert torch.abs(v.grad[:, :, :4, :]).sum() > 0
    assert torch.abs(v.grad[:, :, 4:, :]).sum() == 0
    
    # Zero grads
    v.grad.zero_()
    
    # 3. Test local path
    out_local = local_path(v, window_size=64)
    assert out_local.shape == (B, H, N, D), f"Local path shape mismatch: {out_local.shape}"
    
    # Check first token (should just be itself)
    assert torch.allclose(out_local[:, :, 0, :], v[:, :, 0, :])
    
    # Check 64th token (index 63), should be average of 0..63
    avg_64 = v[:, :, :64, :].mean(dim=-2)
    assert torch.allclose(out_local[:, :, 63, :], avg_64, atol=1e-6)
    
    # Check 128th token (index 127), should be average of 64..127
    avg_128 = v[:, :, 64:128, :].mean(dim=-2)
    assert torch.allclose(out_local[:, :, 127, :], avg_128, atol=1e-6)
    
    out_local.sum().backward()
    assert v.grad is not None and torch.abs(v.grad).sum() > 0
    
    print("All paths passed unit tests!")

if __name__ == "__main__":
    test_paths()
