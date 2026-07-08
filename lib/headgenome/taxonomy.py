"""
This file holds the secret sauce. 
It analyzes the attention heads entirely ZERO-SHOT (no forwarding data).
It uses raw tensor geometry (V/Q ratio and Embed-K lock) to figure out if a head routes semantic info or just looks locally.
"""

import torch
import torch.nn.functional as F

def extract_head_taxonomy(model):
    """
    Scans the model weights and returns a set of (layer_idx, head_idx) that are RETRIEVAL heads.
    Retrieval heads need to stay dense (O(N^2)). Everything else can be windowed (O(N)).
    """
    dense_heads = set()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    
    # Grab the raw embedding weights
    embed_matrix = model.get_input_embeddings().weight.detach()
    
    for layer_idx in range(n_layers):
        try:
            q_proj = model.model.layers[layer_idx].self_attn.q_proj.weight.detach()
            k_proj = model.model.layers[layer_idx].self_attn.k_proj.weight.detach()
            v_proj = model.model.layers[layer_idx].self_attn.v_proj.weight.detach()
        except AttributeError:
            # Fallback for models like Phi-3 that fuse QKV
            qkv = model.model.layers[layer_idx].self_attn.qkv_proj.weight.detach()
            q_proj = qkv[:n_heads * head_dim]
            k_proj = qkv[n_heads * head_dim: n_heads * head_dim + num_kv_heads * head_dim]
            v_proj = qkv[n_heads * head_dim + num_kv_heads * head_dim:]
            
        q_proj = q_proj.view(n_heads, head_dim, -1)
        k_proj = k_proj.view(num_kv_heads, head_dim, -1)
        v_proj = v_proj.view(num_kv_heads, head_dim, -1)
        
        heads_per_kv = n_heads // num_kv_heads
        
        for head_idx in range(n_heads):
            q_w = q_proj[head_idx]
            kv_idx = head_idx // heads_per_kv
            k_w = k_proj[kv_idx]
            v_w = v_proj[kv_idx]
            
            # 1. How deep is this head in the network?
            depth_ratio = layer_idx / n_layers
            
            # 2. Does it push massive vectors compared to its query? (V/Q ratio)
            q_norm = torch.norm(q_w).item()
            v_norm = torch.norm(v_w).item()
            vq_ratio = v_norm / q_norm if q_norm > 0 else 0
            
            # 3. Does it stare at raw syntax embeddings? (Embed-K lock)
            k_embed = F.linear(embed_matrix, k_w)
            k_baseline_norm = torch.norm(k_w).item() * torch.norm(embed_matrix).item()
            embed_k_lock = torch.norm(k_embed).item() / k_baseline_norm if k_baseline_norm > 0 else 0
            
            # The Universal Formula:
            # If it's deep enough, pushes big V vectors, and ignores raw syntax -> It's a semantic router!
            if depth_ratio >= 0.2 and vq_ratio > 1.0 and embed_k_lock < 0.10:
                dense_heads.add((layer_idx, head_idx))
                
    return dense_heads
