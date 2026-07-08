import torch
import time
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
model_id = "Qwen/Qwen2.5-0.5B"

print("Loading model for TTFT Hardware Simulation...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    attn_implementation="sdpa" # Use SDPA for maximum hardware baseline speed
)

results = {"baseline": {}, "router_simulated": {}}
seq_len = 500
step = 500
W = 256 # Local Window size
percentage_local = 263 / 336 # ~78% of heads are local

def measure_ttft(N):
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (1, N)).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids=dummy_input_ids)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        _ = model(input_ids=dummy_input_ids)
    end_event.record()
    torch.cuda.synchronize()
    
    del dummy_input_ids, _
    torch.cuda.empty_cache()
    
    return start_event.elapsed_time(end_event)

print("\nStarting TTFT Benchmarking (Dynamic OOM Scaling)...")

try:
    while True:
        print(f"\n[>] Testing Sequence Length: {seq_len}")
        
        try:
            # 1. Baseline Full Attention (N x N)
            base_time = measure_ttft(seq_len)
            
            # 2. Local Window Attention (N x W)
            # Hardware simulation: In a custom kernel, local heads execute exactly the same 
            # FLOPs as a sequence of length W, executed N/W times, or roughly O(N*W).
            # To empirically simulate the GPU tensor-core latency of computing a 256-window over N tokens,
            # we measure the latency of W tokens, and linearly scale by N/W. 
            # (Because W is constant, O(N*W) scales linearly with N)
            local_time_w = measure_ttft(W) 
            local_time_scaled = local_time_w * (seq_len / W)
            
            # 3. Hybrid Router Simulation
            # 22% of heads compute full N x N (base_time)
            # 78% of heads compute local window N x W (local_time_scaled)
            router_time = ((1.0 - percentage_local) * base_time) + (percentage_local * local_time_scaled)
            
            results["baseline"][seq_len] = base_time
            results["router_simulated"][seq_len] = router_time
            
            print(f"    Baseline TTFT:         {base_time:.2f} ms")
            print(f"    Hybrid Router TTFT:    {router_time:.2f} ms (Simulated Hardware Bound)")
            
        except torch.cuda.OutOfMemoryError:
            print("    [OOM] Hit Out of Memory on RTX 3050!")
            break
            
        gc.collect()
        
        # Save intermediate results
        with open("ttft_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        seq_len += step

except KeyboardInterrupt:
    print("\nBenchmark interrupted by user.")

print("\nSaved ttft_results.json")
