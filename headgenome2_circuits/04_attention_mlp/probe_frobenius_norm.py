import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase2_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_frobenius_norms(model_key="qwen-0.5b"):
    print(f"Loading {model_key}...")
    model, _ = load_model_and_tokenizer(model_key, output_attentions=False, output_hidden_states=False)
    
    # Load the causally-confirmed Counting Heads (Circuit 2)
    with open(os.path.join(OUTPUT_DIR, f"counting_heads_{model_key}.json"), "r") as f:
        counting_data = json.load(f)
        top_counting_heads = counting_data["top_heads"][:4]
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    
    results = []
    
    for head_info in top_counting_heads:
        src_L = head_info["layer"]
        src_H = head_info["head"]
        
        # W_O weight shape: (hidden_size, hidden_size)
        # o_proj applies x @ W^T, so weight is (out_features, in_features)
        # We slice the in_features corresponding to the specific head
        W_O = model.model.layers[src_L].self_attn.o_proj.weight.detach()
        start_idx = src_H * head_dim
        end_idx = start_idx + head_dim
        W_O_head = W_O[:, start_idx:end_idx] # Shape: (hidden_size, head_dim)
        
        # Now iterate over all DOWNSTREAM layers
        for tgt_L in range(src_L + 1, num_layers):
            mlp = model.model.layers[tgt_L].mlp
            # W_gate and W_up weights shape: (intermediate_size, hidden_size)
            W_gate = mlp.gate_proj.weight.detach()
            W_up = mlp.up_proj.weight.detach()
            
            # W_O_head is (hidden_size, head_dim)
            # W_gate is (inter_size, hidden_size)
            # We want || W_gate @ W_O_head ||_F
            
            # 1. Gate Interaction
            inter_gate = torch.matmul(W_gate, W_O_head) # (inter_size, head_dim)
            norm_gate = torch.linalg.matrix_norm(inter_gate, ord='fro').item()
            
            # 2. Up Interaction
            inter_up = torch.matmul(W_up, W_O_head)
            norm_up = torch.linalg.matrix_norm(inter_up, ord='fro').item()
            
            # 3. Fused Interaction (Gate * Up) -> user specified "fused gate-times-up interaction"
            W_fused = W_gate * W_up # Element-wise
            inter_fused = torch.matmul(W_fused, W_O_head)
            norm_fused = torch.linalg.matrix_norm(inter_fused, ord='fro').item()
            
            results.append({
                "source_head": [src_L, src_H],
                "target_mlp": tgt_L,
                "norm_gate": norm_gate,
                "norm_up": norm_up,
                "norm_fused": norm_fused
            })
            
    # Sort results by fused norm
    results = sorted(results, key=lambda x: x["norm_fused"], reverse=True)
    
    out_file = os.path.join(OUTPUT_DIR, f"frobenius_norms_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump({
            "model": model_key,
            "metric": "Frobenius Norm of W_MLP_in @ W_O_head",
            "fused_note": "Fused matrix calculated as W_gate * W_up (element-wise)",
            "rankings": results
        }, f, indent=2)
        
    print(f"Computed Frobenius norms for {len(results)} Head->MLP connections.")
    print("Top 5 Fused Connections:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"Head L{r['source_head'][0]}H{r['source_head'][1]} -> MLP {r['target_mlp']}: Fused={r['norm_fused']:.2f}")
        
if __name__ == "__main__":
    calculate_frobenius_norms("qwen-0.5b")
