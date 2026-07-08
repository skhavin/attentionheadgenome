import torch
import json
import time
import gc

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:
    print("PyTorch 2.5+ is required for flex_attention. Please upgrade.")
    exit(1)

import torch._dynamo
torch._dynamo.config.cache_size_limit = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# Qwen2.5-0.5B Specifications
n_layers = 24
n_heads = 14
head_dim = 64
window_size = 256

print("Compiling FlexAttention Kernel (This may take a minute)...")

# Define the Universal Router rule
def hybrid_mask(b, h, q_idx, kv_idx):
    # In Triton, we return boolean masks
    # For heads that are local, enforce W=256 and causal
    local_cond = (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size)
    # Also keep sinks (first 4 tokens)
    local_cond = local_cond | (kv_idx < 4)
    
    # Dense heads just need causal
    dense_cond = (q_idx >= kv_idx)
    
    # We simulate 78% local heads by using the head index
    # (Since there are 14 heads, 11 heads are local, 3 are dense)
    is_local_head = h < 11 
    
    return torch.where(is_local_head, local_cond, dense_cond)

flex_compiled = torch.compile(flex_attention)

results = {"baseline_sdpa": {}, "compiled_flex_hybrid": {}}
seq_len = 500
step = 500

print("\nStarting Isolated Kernel Benchmark (TTFT Estimation)...")

try:
    while seq_len <= 8000:
        print(f"\n[>] Testing Sequence Length: {seq_len}")
        
        q = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        
        # PyTorch 2.5+ uses create_block_mask which must be computed per-sequence length
        block_mask = create_block_mask(hybrid_mask, B=1, H=n_heads, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        
        # --- BASELINE (Native C++ SDPA) ---
        try:
            # Warmup
            for _ in range(3):
                torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(n_layers): # Simulate full network pass
                torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            end_event.record()
            torch.cuda.synchronize()
            
            base_time = start_event.elapsed_time(end_event)
            results["baseline_sdpa"][seq_len] = base_time
            print(f"    Dense SDPA TTFT (24 Layers): {base_time:.2f} ms")
        except torch.cuda.OutOfMemoryError:
            print("    [OOM] Baseline Hit Out of Memory!")
            break
            
        # --- COMPILED FLEX HYBRID ROUTER ---
        try:
            # Warmup
            for _ in range(3):
                flex_compiled(q, k, v, block_mask=block_mask)
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(n_layers): # Simulate full network pass
                flex_compiled(q, k, v, block_mask=block_mask)
            end_event.record()
            torch.cuda.synchronize()
            
            flex_time = start_event.elapsed_time(end_event)
            results["compiled_flex_hybrid"][seq_len] = flex_time
            print(f"    Compiled Flex TTFT (24 Layers): {flex_time:.2f} ms")
        except torch.cuda.OutOfMemoryError:
            print("    [OOM] Flex Hit Out of Memory!")
            break
            
        del q, k, v, block_mask
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save results
        with open("flex_ttft_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        seq_len += step

except KeyboardInterrupt:
    print("\nBenchmark interrupted.")

print("\nSaved flex_ttft_results.json")
