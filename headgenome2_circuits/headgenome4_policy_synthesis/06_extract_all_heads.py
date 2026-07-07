import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

MODELS = {
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B": "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B"
}

PROMPT_PAIRS = [
    ("The capital of France is Paris.", " The capital of France is"),
    ("The speed of light is 299792458 meters per second.", " The speed of light is"),
    ("Shakespeare was born in Stratford-upon-Avon.", " Shakespeare was born in"),
    ("Water boils at 100 degrees Celsius.", " Water boils at"),
    ("The Eiffel Tower is located in Paris.", " The Eiffel Tower is located in"),
] # Reduced prompts for speed

def extract_all_heads():
    os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
    
    for model_key, model_id in MODELS.items():
        print(f"\nProcessing {model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda", attn_implementation="eager")
        model.eval()
        
        d_model = model.config.hidden_size
        n_heads = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_heads)
        n_layers = model.config.num_hidden_layers
        head_dim = d_model // n_heads
        
        results = []
        
        for context, query in tqdm(PROMPT_PAIRS, desc=f"Extracting {model_key}"):
            prompt = context + query
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
            hidden_states = outputs.hidden_states
            seq_len = hidden_states[0].shape[1]
            
            C = [hidden_states[0].squeeze(0).cpu()]
            for l in range(1, len(hidden_states)):
                C.append((hidden_states[l] - hidden_states[l-1]).squeeze(0).cpu())
                
            for L in range(n_layers):
                layer_module = model.model.layers[L]
                q_proj = layer_module.self_attn.q_proj.weight.data.cpu()
                k_proj = layer_module.self_attn.k_proj.weight.data.cpu()
                x_in = hidden_states[L].squeeze(0).cpu()
                
                for H in range(n_heads):
                    wq = q_proj[H * head_dim : (H + 1) * head_dim, :].T
                    kv_group = H // (n_heads // n_kv) if n_kv != n_heads else H
                    wk = k_proj[kv_group * head_dim : (kv_group + 1) * head_dim, :].T
                    
                    Q = x_in @ wq
                    K = x_in @ wk
                    scores = (Q @ K.T) / np.sqrt(head_dim)
                    
                    mask = torch.tril(torch.ones(seq_len, seq_len))
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                    
                    query_idx = seq_len - 1
                    attn_probs = torch.softmax(scores[query_idx], dim=-1)
                    attn_probs_no_self = attn_probs.clone()
                    attn_probs_no_self[query_idx] = 0
                    max_target_idx = torch.argmax(attn_probs_no_self).item()
                    
                    q_comps = torch.stack([c[query_idx] @ wq for c in C[:L+1]])
                    k_comps = torch.stack([c[max_target_idx] @ wk for c in C[:L+1]])
                    
                    interaction_matrix = (q_comps @ k_comps.T) / np.sqrt(head_dim)
                    q_layer_contribs = interaction_matrix.sum(dim=1)
                    k_layer_contribs = interaction_matrix.sum(dim=0)
                    
                    results.append({
                        "layer": L,
                        "head": H,
                        "top_q_layer": torch.argmax(q_layer_contribs).item(),
                        "top_k_layer": torch.argmax(k_layer_contribs).item(),
                        "q_embed_contrib": q_layer_contribs[0].item(),
                        "k_embed_contrib": k_layer_contribs[0].item(),
                        "total_score": interaction_matrix.sum().item(),
                    })
                    
        df = pd.DataFrame(results)
        agg = df.groupby(["layer", "head"]).agg({
            "top_q_layer": lambda x: x.mode()[0] if not x.empty else 0,
            "top_k_layer": lambda x: x.mode()[0] if not x.empty else 0,
            "k_embed_contrib": "mean",
            "total_score": "mean"
        }).reset_index()
        
        os.makedirs("outputs/phase1", exist_ok=True)
        out_path = f"outputs/phase1/component_attribution_all_{model_key}.csv"
        agg.to_csv(out_path, index=False)
        print(f"Saved complete attribution profile to {out_path}")
        
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    extract_all_heads()
