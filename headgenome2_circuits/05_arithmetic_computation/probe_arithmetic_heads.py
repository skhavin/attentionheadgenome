import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

DATASET_PATH = "headgenome2_circuits/datasets/arithmetic.json"
OUTPUT_DIR = "outputs/phase2_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_operand_indices(input_ids, tokenizer, x, y):
    tokens = [tokenizer.decode([tok]) for tok in input_ids[0]]
    x_str = str(x)
    y_str = str(y)
    
    x_idx, y_idx = -1, -1
    for i, tok in enumerate(tokens):
        if x_str in tok and x_idx == -1:
            x_idx = i
        elif y_str in tok and y_idx == -1:
            y_idx = i
            
    return x_idx, y_idx

def probe_arithmetic(model_key="qwen-0.5b", num_samples=100):
    print(f"Loading {model_key}...")
    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=True)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)[:num_samples]
        
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    head_masses = torch.zeros((num_layers, num_heads), device=model.device)
    valid_count = 0
    
    for item in tqdm(dataset, desc="Probing Arithmetic"):
        prompt = item["prompt"]
        x, y = item["x"], item["y"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        
        x_idx, y_idx = find_operand_indices(input_ids, tokenizer, x, y)
        if x_idx == -1 or y_idx == -1:
            continue
            
        valid_count += 1
        with torch.no_grad():
            outputs = model(**inputs)
            
        attentions = outputs.attentions # (num_layers) tuple of (batch, heads, seq, seq)
        
        # We look at the attention from the very last token (the generation step for the sum)
        last_tok_idx = input_ids.shape[1] - 1
        
        for l in range(num_layers):
            attn = attentions[l][0] # (heads, seq, seq)
            # Mass on operands
            mass = attn[:, last_tok_idx, x_idx] + attn[:, last_tok_idx, y_idx]
            head_masses[l] += mass
            
    head_masses /= valid_count
    
    # Rank heads
    flat_masses = head_masses.flatten().cpu().numpy()
    top_indices = np.argsort(flat_masses)[::-1]
    
    results = []
    for idx in top_indices[:20]:
        l = idx // num_heads
        h = idx % num_heads
        mass = flat_masses[idx]
        results.append({
            "layer": int(l),
            "head": int(h),
            "operand_attention_mass": float(mass)
        })
        
    print(f"\nTop 10 Arithmetic Heads ({model_key}):")
    for i in range(10):
        print(f"L{results[i]['layer']}H{results[i]['head']} - Mass: {results[i]['operand_attention_mass']:.3f}")
        
    out_file = os.path.join(OUTPUT_DIR, f"arithmetic_heads_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump({
            "model": model_key,
            "metric": "attention_mass_on_operands",
            "top_heads": results
        }, f, indent=2)
        
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    probe_arithmetic("qwen-0.5b")
