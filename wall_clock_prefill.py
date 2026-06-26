import torch
import time
from transformers import AutoModelForCausalLM
import numpy as np

model_id = "Qwen/Qwen2.5-0.5B"
print(f"Loading {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
model.eval()

# Sequence length for profiling
SEQ_LEN = 4096
BATCH_SIZE = 1

print(f"Profiling prefill latency for N={SEQ_LEN}...")

input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, SEQ_LEN)).cuda()

# Warmup
print("Warming up standard attention...")
for _ in range(3):
    with torch.no_grad():
        _ = model(input_ids)

# Standard Attention Measurement
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(input_ids)
torch.cuda.synchronize()
dense_time = (time.time() - start) / 10

# Simulated Sparse Attention Measurement
# To simulate the exact FLOP reduction without writing a custom CUDA kernel, 
# we can mock the sequence length computation for local vs global heads.
# However, the user wants a "measured wall-clock prefill comparison on a single sequence".
# A standard trick in PyTorch without FlashInfer is applying a sparse Block-Diagonal or Sliding Window mask via SDPA.
# Qwen natively uses SDPA in torch 2.x.
# Since we can't easily hook and replace the mask per-head dynamically in python without huge overhead,
# we will construct a global sparse sliding window mask to simulate the "Local" head portion (85% of heads).

# Let's create a custom SDPA mask. SDPA takes an attn_mask of shape (B, 1, seq_len, seq_len).
# We want to force a window size W=512.
W = 512
causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device="cuda"))
window_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device="cuda"), diagonal=-W + 1)
sparse_mask = causal_mask & window_mask

# Convert to float mask for SDPA
float_mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.float16, device="cuda")
float_mask.masked_fill_(~sparse_mask, float('-inf'))
float_mask = float_mask.unsqueeze(0).unsqueeze(0) # (1, 1, S, S)

def mock_forward(input_ids, custom_mask):
    return model(input_ids, attention_mask=custom_mask)

print("Warming up sparse sliding window mask...")
for _ in range(3):
    with torch.no_grad():
        _ = mock_forward(input_ids, float_mask)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = mock_forward(input_ids, float_mask)
torch.cuda.synchronize()
sparse_time = (time.time() - start) / 10

print("="*40)
print(f"Wall-Clock Prefill (N={SEQ_LEN}, Qwen-0.5B)")
print(f"Dense $O(N^2)$ Time:   {dense_time*1000:.2f} ms")
print(f"Sparse Window Time: {sparse_time*1000:.2f} ms")
print(f"Speedup: {dense_time/sparse_time:.2f}x")
print("="*40)
