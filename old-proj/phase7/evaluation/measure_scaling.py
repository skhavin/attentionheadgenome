import os
import sys
import time
import torch
import json
import argparse
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase7.moe.moe_patcher import MoEPatcher

def measure_time(model, input_ids, num_iters=10):
    # warmup
    for _ in range(2):
        with torch.no_grad():
            model(input_ids)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    
    end = time.perf_counter()
    return ((end - start) / num_iters) * 1000 # ms per forward pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint", default="checkpoints/latest-qwen-fixed")
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--output", default="outputs/phase7/scaling_curve.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    seq_lens = [256, 512, 1024, 2048, 4096]
    
    results = {"seq_lens": seq_lens, "baseline_ms": [], "moe_ms": []}
    
    print("Measuring baseline...")
    for seq_len in seq_lens:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(device)
        try:
            ms = measure_time(model, input_ids, num_iters=10)
            print(f"  Baseline len {seq_len}: {ms:.2f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at len {seq_len}")
                ms = None
            else:
                raise e
        results["baseline_ms"].append(ms)

    print("Installing MoEPatcher...")
    patcher = MoEPatcher(model, hard_routing=True)
    
    ckpt_path = os.path.join(args.checkpoint, f"stage{args.stage}_routers.pt")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        for name, router in patcher.routers.items():
            if name in state_dict:
                router.load_state_dict(state_dict[name])
        print(f"Loaded router weights from {ckpt_path}")
    else:
        print(f"WARNING: No router weights found at {ckpt_path}. Using uniform init.")
        
    for router in patcher.routers.values():
        router.to(model.dtype)
        
    print("Measuring MoE...")
    for i, seq_len in enumerate(seq_lens):
        if results["baseline_ms"][i] is None:
            results["moe_ms"].append(None)
            continue
            
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(device)
        try:
            ms = measure_time(model, input_ids, num_iters=10)
            print(f"  MoE len {seq_len}: {ms:.2f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at len {seq_len}")
                ms = None
            else:
                raise e
        results["moe_ms"].append(ms)
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
