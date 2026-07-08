import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

q = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.bfloat16)

def window_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

try:
    # PyTorch 2.6+ uses create_block_mask
    block_mask = create_block_mask(window_mask, B=1, H=1, Q_LEN=128, KV_LEN=128)
    
    flex_attn = torch.compile(flex_attention)
    out = flex_attn(q, k, v, block_mask=block_mask)
    print('FlexAttention Compile and Execution Success with block_mask!')
except Exception as e:
    print('FlexAttention failed to compile or run:')
    print(e)
