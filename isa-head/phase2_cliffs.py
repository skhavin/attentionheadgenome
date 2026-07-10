import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    gt = sum(1 for i in x for j in y if i > j)
    lt = sum(1 for i in x for j in y if i < j)
    return (gt - lt) / (n1 * n2)

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, torch_dtype=torch.bfloat16, device_map="cuda")

with open("dataset_confirmation_20.json", "r", encoding="utf-8") as f:
    prompts = [item for item in json.load(f) if item["task_type"] == "niah"]

with open("phase2_retrieval_heads.json", "r") as f:
    registered_heads = json.load(f)

target_vals = []
uniform_vals = []
distance_vals = []

for item in prompts:
    tokens = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
    pwd_tokens = tokenizer(item["password"], add_special_tokens=False).input_ids
    input_ids = tokens.input_ids[0].tolist()
    
    target_idx = -1
    for i in range(len(input_ids) - len(pwd_tokens)):
        if input_ids[i:i+len(pwd_tokens)] == pwd_tokens:
            target_idx = i + len(pwd_tokens) - 1
            break
            
    Q_idx = len(input_ids) - 1
    with torch.no_grad():
        outputs = model(**tokens)
        
    for rh in registered_heads:
        l, h = rh["layer"], rh["head"]
        attn = outputs.attentions[l][0]
        
        target_vals.append(attn[h, Q_idx, target_idx].item())
        uniform_vals.append(1.0 / (Q_idx + 1))
        
        if Q_idx - 1 >= 0 and target_idx - 1 >= 0:
            distance_vals.append(attn[h, Q_idx - 1, target_idx - 1].item())
        else:
            distance_vals.append(1.0 / (Q_idx + 1))

delta_uni = cliffs_delta(target_vals, uniform_vals)
delta_dist = cliffs_delta(target_vals, distance_vals)

print(f"Cliff's Delta (Target vs Uniform): {delta_uni:.4f}")
print(f"Cliff's Delta (Target vs Positional): {delta_dist:.4f}")
