import json
import torch
import os
import sys
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

DATASET_PATH = "headgenome2_circuits/datasets/copy_uuids.json"
OUTPUT_DIR = "outputs/phase2_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_subsequence(sequence, subseq):
    """Finds the starting index of subseq in sequence."""
    for i in range(len(sequence) - len(subseq) + 1):
        if sequence[i:i+len(subseq)] == subseq:
            return i
    return -1

def run_probe(model_key="qwen-0.5b"):
    model, tokenizer = load_model_and_tokenizer(model_key)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # We will accumulate the attention mass each head directs to the target UUID
    accumulated_mass = torch.zeros((num_layers, num_heads), device=model.device)
    valid_prompts = 0
    
    for item in dataset:
        prompt = item["prompt"]
        target = item["target"]
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        
        # We need to find where the target_ids are located inside input_ids
        input_list = input_ids[0].tolist()
        
        # UUIDs might be tokenized weirdly when preceded by a space. 
        # Let's do a fuzzy subseq search or encode specifically.
        # A simpler way: target is explicitly in the prompt after "User Request ID: "
        prefix_ids = tokenizer.encode("User Request ID: ", add_special_tokens=False)
        start_idx = find_subsequence(input_list, target_ids)
        
        # If strict token match fails due to prefix spacing, we find it manually.
        if start_idx == -1:
            # Fallback: find prefix and assume target follows
            pref_idx = find_subsequence(input_list, prefix_ids)
            if pref_idx != -1:
                start_idx = pref_idx + len(prefix_ids)
            else:
                continue # Skip if we can't reliably find the target tokens
                
        end_idx = start_idx + len(target_ids)
        
        with torch.no_grad():
            outputs = model(input_ids)
            attentions = outputs.attentions # Tuple of (batch, heads, seq, seq)
            
            # Look at the attention from the very last token in the sequence (the prompt generation point)
            seq_len = input_ids.shape[1]
            last_token_idx = seq_len - 1
            
            for L in range(num_layers):
                # (batch, heads, seq, seq) -> [0, :, last_token_idx, start_idx:end_idx]
                attn_layer = attentions[L][0] 
                target_mass = attn_layer[:, last_token_idx, start_idx:end_idx].sum(dim=-1) # sum across target tokens
                accumulated_mass[L] += target_mass
                
        valid_prompts += 1
        
    print(f"Processed {valid_prompts}/{len(dataset)} valid prompts.")
    
    avg_mass = (accumulated_mass / valid_prompts).cpu().numpy()
    
    # Identify top heads
    flat_indices = np.argsort(avg_mass.flatten())[::-1]
    top_heads = []
    for i in range(10): # Top 10 heads
        idx = flat_indices[i]
        layer = idx // num_heads
        head = idx % num_heads
        val = avg_mass[layer, head]
        top_heads.append({"layer": int(layer), "head": int(head), "mass": float(val)})
        print(f"Top {i+1}: L{layer}H{head} - Mass: {val:.3f}")
        
    # Save output
    output_file = os.path.join(OUTPUT_DIR, f"copy_heads_{model_key}.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": model_key,
            "metric": "Average attention mass to exact UUID tokens",
            "samples": valid_prompts,
            "top_heads": top_heads,
            "full_matrix": avg_mass.tolist()
        }, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_probe("qwen-0.5b")
