import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase7.kernel_attention import FrozenKernelAttentionPatcher
import numpy as np

def verify_exactness():
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)
    model.eval()

    text = "The cat sat on the mat and looked at the bird outside the window."
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Get full exact attention output (hidden states)
        exact_outputs = model(**inputs, output_hidden_states=True)
        exact_hidden = exact_outputs.hidden_states[-1] # Last layer hidden states
        
        # Now patch and get approx output
        patcher = FrozenKernelAttentionPatcher(model, method="frozen_kernel")
        approx_outputs = model(**inputs, output_hidden_states=True)
        approx_hidden = approx_outputs.hidden_states[-1]
        
        patcher.restore()

    diff = (exact_hidden - approx_hidden).abs()
    l_inf = diff.max().item()
    l_mean = diff.mean().item()

    print(f"L_inf norm between exact and frozen_kernel hidden states: {l_inf}")
    print(f"Mean absolute difference: {l_mean}")
    
    # Also check logits
    exact_logits = exact_outputs.logits
    approx_logits = approx_outputs.logits
    logit_diff = (exact_logits - approx_logits).abs()
    print(f"Logits L_inf diff: {logit_diff.max().item()}")
    
if __name__ == "__main__":
    verify_exactness()
