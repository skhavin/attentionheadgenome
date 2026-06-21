import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase7.audit_heads import extract_V, detect_arch, iter_attn_layers

model = AutoModelForCausalLM.from_pretrained('gpt2-medium', torch_dtype=torch.float16, device_map='auto')
arch = detect_arch(model)
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]

V = extract_V(model, input_ids, layer_idx=0, head_idx=0, arch=arch, device=model.device)
print(f"V is {V}")
