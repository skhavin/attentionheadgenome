import torch
import sys, os

sys.path.append("phase7")
from regime_detector import RegimeSwitchingPatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "gpt2-medium"
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tier1 = [(4, 11, "local")]
    patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1, tier2_heads=[])
    
    text = "The quick brown fox jumps over the lazy dog"
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    # Original output (without patch)
    with torch.no_grad():
        out_patched = model(ids).logits
        
    patcher.restore()
    
    with torch.no_grad():
        out_orig = model(ids).logits
        
    diff = (out_patched - out_orig).abs().max().item()
    print("Maximum logit difference between original and patched:", diff)

if __name__ == "__main__":
    main()
