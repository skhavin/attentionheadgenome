import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "The quick brown fox jumps over the lazy dog. " * 5
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Let's inspect the hidden states norm at each layer
    print("Layer-by-layer hidden state mean norm:")
    for l, hs in enumerate(outputs.hidden_states):
        # hs shape: [B, N, hidden_size]
        norm = hs.norm(dim=-1).mean().item()
        std = hs.std().item()
        print(f"  Layer {l:2d}: Mean Norm = {norm:9.3f}, std = {std:9.5f}")

if __name__ == "__main__":
    main()
