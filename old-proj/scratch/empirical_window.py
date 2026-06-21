import torch
import numpy as np

path = "d:/PROJECTS/webstromprojects/supertransformers/outputs/phase0/attention_0.pt"
data = torch.load(path, weights_only=False)
layer0_attn = data["attentions"][0][0] # (heads, seq, seq)
head0_attn = layer0_attn[0] # (seq, seq)

seq_len = head0_attn.shape[0]

W_empirical = []
for q in range(seq_len):
    row = head0_attn[q, :q+1]
    # sort backwards from q to 0
    row_reversed = row.flip(dims=[0])
    cum_mass = row_reversed.cumsum(dim=0)
    
    # find first idx where cum_mass > 0.99
    w_idx = (cum_mass > 0.99).nonzero(as_tuple=True)[0]
    if len(w_idx) > 0:
        W_empirical.append(w_idx[0].item() + 1)
    else:
        W_empirical.append(q + 1)
        
print(f"Empirical window for Layer 0 Head 0 to cover 99% mass: Mean={np.mean(W_empirical):.2f}, Max={np.max(W_empirical)}, P90={np.percentile(W_empirical, 90):.0f}")
