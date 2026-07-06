import json
import torch
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

DATASET_PATH = "headgenome2_circuits/datasets/wikitext_random.json"
OUTPUT_DIR = "outputs/phase2_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_entropy(attn_matrix):
    """Computes average entropy of attention distributions across sequence length."""
    # Convert to float32 to avoid float16 1e-9 becoming exactly 0
    attn_matrix = attn_matrix.float()
    eps = 1e-9
    # Entropy H = -sum(p * log(p))
    entropy_per_token = -torch.sum(attn_matrix * torch.log(attn_matrix + eps), dim=-1)
    return entropy_per_token.mean().item()

def profile_heads(model_key="qwen-0.5b"):
    model, tokenizer = load_model_and_tokenizer(model_key)
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    accumulated_entropy = torch.zeros((num_layers, num_heads), device=model.device)
    valid_prompts = 0
    
    for item in dataset:
        prompt = item["text"]
        # truncate to 256 tokens for speed
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)
        
        if input_ids.shape[1] < 10:
            continue
            
        with torch.no_grad():
            outputs = model(input_ids)
            attentions = outputs.attentions # Tuple of (batch, heads, seq, seq)
            
            for L in range(num_layers):
                attn_layer = attentions[L][0] # (heads, seq, seq)
                for H in range(num_heads):
                    accumulated_entropy[L, H] += compute_entropy(attn_layer[H])
                    
        valid_prompts += 1
        
    avg_entropy = (accumulated_entropy / valid_prompts).cpu().numpy()
    
    output_file = os.path.join(OUTPUT_DIR, f"head_entropy_{model_key}.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": model_key,
            "metric": "Average Attention Entropy (Wikitext)",
            "samples": valid_prompts,
            "entropy_matrix": avg_entropy.tolist()
        }, f, indent=2)
    print(f"Entropy profile saved to {output_file}")

if __name__ == "__main__":
    profile_heads("qwen-0.5b")
