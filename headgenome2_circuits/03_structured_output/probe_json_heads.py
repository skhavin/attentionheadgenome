import json
import torch
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

DATASET_PATH = "headgenome2_circuits/datasets/json_brackets.json"
OUTPUT_DIR = "outputs/phase2_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_probe(model_key="qwen-0.5b"):
    model, tokenizer = load_model_and_tokenizer(model_key)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # We will accumulate the attention mass each head directs to the open braces '{'
    accumulated_mass = torch.zeros((num_layers, num_heads), device=model.device)
    valid_prompts = 0
    
    for item in dataset:
        prompt = item["prompt"]
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        input_list = input_ids[0].tolist()
        
        # Find indices of tokens containing '{'
        brace_indices = []
        for i, tok_id in enumerate(input_list):
            tok_str = tokenizer.decode([tok_id])
            if '{' in tok_str:
                brace_indices.append(i)
                
        if not brace_indices:
            continue
            
        with torch.no_grad():
            outputs = model(input_ids)
            attentions = outputs.attentions 
            
            seq_len = input_ids.shape[1]
            last_token_idx = seq_len - 1
            
            for L in range(num_layers):
                attn_layer = attentions[L][0] 
                
                # Sum attention from the last token to all identified brace tokens
                mass = 0.0
                for idx in brace_indices:
                    mass += attn_layer[:, last_token_idx, idx]
                
                accumulated_mass[L] += mass
                
        valid_prompts += 1
        
    avg_mass = (accumulated_mass / valid_prompts).cpu().numpy()
    
    # Identify top heads
    flat_indices = np.argsort(avg_mass.flatten())[::-1]
    top_heads = []
    for i in range(10):
        idx = flat_indices[i]
        layer = idx // num_heads
        head = idx % num_heads
        val = avg_mass[layer, head]
        top_heads.append({"layer": int(layer), "head": int(head), "mass": float(val)})
        print(f"Top {i+1}: L{layer}H{head} - Mass: {val:.3f}")
        
    output_file = os.path.join(OUTPUT_DIR, f"json_heads_{model_key}.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": model_key,
            "metric": "Average attention mass to open braces '{'",
            "samples": valid_prompts,
            "top_heads": top_heads
        }, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_probe("qwen-0.5b")
