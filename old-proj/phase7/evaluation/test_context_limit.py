import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase7.moe.moe_patcher import MoEPatcher

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 1. Test get_activation_stats() logic
    print("\n--- Testing Activation Stats ---")
    patcher = MoEPatcher(model)
    patcher.reset_activation_stats()
    
    # Force full attention on all layers
    for layer_idx in range(len(patcher.routers)):
        patcher.force_full_attention(layer_idx)
        
    # Dummy forward pass
    dummy_input = torch.randint(0, 1000, (1, 128)).to(device)
    with torch.no_grad():
        model(dummy_input)
        
    stats = patcher.get_activation_stats()
    print(f"Stats with forced full attention (Sink, Local, Rec, Full): {stats}")
    if sum(stats) == 0:
        print("Note: forced full attention bypasses the router forward entirely, so stats might be 0.0.")
    else:
        print(f"Sum: {sum(stats)}%")
        
    # Let's test with normal routing (unforced, but active_routing_layer = 9999 to simulate fallback or uniform)
    patcher.full_attention_only_layers.clear()
    patcher.reset_activation_stats()
    patcher.active_routing_layer = 9999 # Causes probs to be [0, 0, 0, 1] internally for progressive fallback
    with torch.no_grad():
        model(dummy_input)
    stats2 = patcher.get_activation_stats()
    print(f"Stats with active_routing_layer fallback (should be 100% full): {stats2}")
    print(f"Sum: {sum(stats2)}%")
    
    # Let's test with uniform distribution (no hard routing)
    patcher.active_routing_layer = -1
    patcher.hard_routing = False
    patcher.reset_activation_stats()
    # The routers are randomly initialized (weights near zero), so they should output approx 25% each
    with torch.no_grad():
        model(dummy_input)
    stats3 = patcher.get_activation_stats()
    print(f"Stats with soft routing (untrained routers, uniform): {stats3}")
    print(f"Sum: {sum(stats3):.2f}%")
    
    patcher.restore()
    del patcher
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Test max context length support
    print("\n--- Testing GPU Context Length Support ---")
    lengths = [4096, 8192, 16384, 32768, 65536]
    supported_length = 0
    
    for length in lengths:
        try:
            print(f"Testing seq_len={length}...")
            # We don't actually generate 32k tokens because it's too slow.
            # We just do a single forward pass with a sequence of `length` tokens to check VRAM.
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, length)).to(device)
            with torch.no_grad():
                _ = model(input_ids)
                
            print(f"  Success! VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            supported_length = length
            del input_ids
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at {length} tokens. VRAM limit reached.")
                torch.cuda.empty_cache()
                break
            else:
                print(f"  Error: {e}")
                break
                
    print(f"\nMax supported context length tested: {supported_length}")

if __name__ == "__main__":
    main()
