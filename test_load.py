import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
try:
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    print("Success! Model loaded.")
    
    # Test prompt
    prompt = "Alice went to the store. Bob went to the store. Alice went to the"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=2)
    print("Test generation:", tokenizer.decode(out[0]))
except Exception as e:
    print("Error:", e)
