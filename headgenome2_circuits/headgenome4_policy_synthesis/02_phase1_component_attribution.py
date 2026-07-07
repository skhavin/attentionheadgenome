import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B": "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
}

PROMPT_PAIRS = [
    ("The capital of France is Paris.", " The capital of France is"),
    ("The speed of light is 299792458 meters per second.", " The speed of light is"),
    ("Shakespeare was born in Stratford-upon-Avon.", " Shakespeare was born in"),
    ("Water boils at 100 degrees Celsius.", " Water boils at"),
    ("The Eiffel Tower is located in Paris.", " The Eiffel Tower is located in"),
    ("Mount Everest is the tallest mountain on Earth.", " Mount Everest is the tallest"),
    ("The chemical formula for water is H2O.", " The chemical formula for water is"),
    ("Leonardo da Vinci painted the Mona Lisa.", " Leonardo da Vinci painted the"),
    ("The Great Wall of China was built over many centuries.", " The Great Wall of China was"),
    ("Albert Einstein developed the theory of relativity.", " Albert Einstein developed the theory of"),
    ("Oxygen has the atomic number 8.", " Oxygen has the atomic number"),
    ("The Amazon River flows through Brazil.", " The Amazon River flows through"),
    ("The human body has 206 bones.", " The human body has"),
    ("Isaac Newton discovered gravity.", " Isaac Newton discovered"),
    ("The Pacific Ocean is the largest ocean on Earth.", " The Pacific Ocean is the"),
    ("DNA stands for deoxyribonucleic acid.", " DNA stands for"),
    ("Rome is the capital of Italy.", " Rome is the capital of"),
    ("The Berlin Wall fell in 1989.", " The Berlin Wall fell in"),
    ("The Pythagorean theorem states that a squared plus b squared equals c squared.", " The Pythagorean theorem states"),
    ("Photosynthesis converts sunlight into chemical energy.", " Photosynthesis converts sunlight into"),
]

def load_canonical_labels(model_name):
    path = "outputs/canonical_labels.json"
    with open(path, "r") as f:
        data = json.load(f)
    if model_name not in data["models"]:
        return {}
    
    heads = data["models"][model_name]["heads"]
    # Group by label
    grouped = {"induction": [], "retrieval": [], "sink": [], "local": []}
    for k, v in heads.items():
        grouped[v["label"]].append((v["layer"], v["head_idx"]))
    
    # Subsample local heads to save time (max 10)
    import random
    random.seed(42)
    if len(grouped["local"]) > 10:
        grouped["local"] = random.sample(grouped["local"], 10)
        
    return grouped

def extract_head_weights(model, layer_idx, head_idx, d_model, n_heads, n_kv):
    layer = model.model.layers[layer_idx]
    q_proj = layer.self_attn.q_proj.weight.data
    k_proj = layer.self_attn.k_proj.weight.data
    
    head_dim = d_model // n_heads
    # Extract specific head
    wq = q_proj[head_idx * head_dim : (head_idx + 1) * head_dim, :]
    
    # Handle GQA
    kv_group = head_idx // (n_heads // n_kv) if n_kv != n_heads else head_idx
    wk = k_proj[kv_group * head_dim : (kv_group + 1) * head_dim, :]
    
    # Return as (d_model, head_dim) for x @ W 
    return wq.T, wk.T

def apply_rotary_pos_emb(x, position_ids, model, head_dim):
    # Retrieve RoPE module
    # Implementation depends on the model (Llama vs Qwen)
    # This is a simplification; a full RoPE application is complex to write zero-shot.
    # To keep it exact, we can pull Q and K vectors directly from the model's forward pass
    # instead of recomputing them with RoPE. 
    pass

def run_attribution():
    os.makedirs("outputs/phase1", exist_ok=True)
    os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
    
    for model_key, model_id in MODELS.items():
        print(f"\nProcessing {model_key}...")
        head_groups = load_canonical_labels(model_key)
        total_heads = sum(len(v) for v in head_groups.values())
        if total_heads == 0:
            print(f"Skipping {model_key} - no labels found.")
            continue
            
        print(f"Tracking {total_heads} heads (Induction: {len(head_groups['induction'])}, Retrieval: {len(head_groups['retrieval'])}, Sink: {len(head_groups['sink'])}, Local: {len(head_groups['local'])})")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
        model.eval()
        
        d_model = model.config.hidden_size
        n_heads = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_heads)
        head_dim = d_model // n_heads
        
        results = []
        
        for context, query in tqdm(PROMPT_PAIRS):
            prompt = context + query
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                # We need output_hidden_states to get the residual stream
                outputs = model(**inputs, output_hidden_states=True)
                
            hidden_states = outputs.hidden_states
            seq_len = hidden_states[0].shape[1]
            
            # C[i] = contribution of layer i to the residual stream
            C = [hidden_states[0].squeeze(0).cpu()]
            for l in range(1, len(hidden_states)):
                C.append((hidden_states[l] - hidden_states[l-1]).squeeze(0).cpu())
                
            for label, heads in head_groups.items():
                for L, H in heads:
                    wq, wk = extract_head_weights(model, L, H, d_model, n_heads, n_kv)
                    wq = wq.cpu()
                    wk = wk.cpu()
                    
                    x_in = hidden_states[L].squeeze(0).cpu()
                    
                    # Because Qwen and Llama use RoPE, recomputing Q and K exactly is hard
                    # without accessing their internal rope embeddings.
                    # We will approximate the component contributions pre-RoPE.
                    # Or we can decompose the interaction and the RoPE effects will be part of the score.
                    # Since RoPE is a rotation matrix R, Q_rope = Q @ R, K_rope = K @ R
                    # Q_rope @ K_rope.T = Q @ R @ R^T @ K.T = Q @ K.T (if relative pos = 0)
                    # But if relative pos > 0, it's Q @ R_m @ R_n^T @ K.T = Q @ R_{m-n} @ K.T
                    # We will omit RoPE in the attribution for now, which gives us the "content" contribution.
                    
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
                    max_attn_weight = attn_probs[max_target_idx].item()
                    
                    if label in ["retrieval", "sink"] and H == heads[0][1]: # Just print for the first head of this class to avoid spam
                        target_token = tokenizer.decode(inputs.input_ids[0, max_target_idx])
                        print(f"    [{label.upper()} L{L}H{H}] Looked at token: '{target_token}' (idx {max_target_idx}) with weight {max_attn_weight:.2f}")
                    
                    q_comps = torch.stack([c[query_idx] @ wq for c in C[:L+1]])
                    k_comps = torch.stack([c[max_target_idx] @ wk for c in C[:L+1]])
                    
                    interaction_matrix = (q_comps @ k_comps.T) / np.sqrt(head_dim)
                    
                    q_layer_contribs = interaction_matrix.sum(dim=1)
                    k_layer_contribs = interaction_matrix.sum(dim=0)
                    
                    top_q_layer = torch.argmax(q_layer_contribs).item()
                    top_k_layer = torch.argmax(k_layer_contribs).item()
                    
                    results.append({
                        "model": model_key,
                        "layer": L,
                        "head": H,
                        "label": label,
                        "max_target_idx": max_target_idx,
                        "max_attn_weight": max_attn_weight,
                        "top_q_layer": top_q_layer,
                        "top_k_layer": top_k_layer,
                        "q_embed_contrib": q_layer_contribs[0].item(),
                        "k_embed_contrib": k_layer_contribs[0].item(),
                        "total_score": interaction_matrix.sum().item(),
                    })
                    
        df = pd.DataFrame(results)
        agg = df.groupby(["layer", "head", "label"]).agg({
            "max_attn_weight": "mean",
            "top_q_layer": lambda x: x.mode()[0] if not x.empty else 0,
            "top_k_layer": lambda x: x.mode()[0] if not x.empty else 0,
            "q_embed_contrib": "mean",
            "k_embed_contrib": "mean",
            "total_score": "mean"
        }).reset_index()
        
        out_path = f"outputs/phase1/component_attribution_{model_key}.csv"
        agg.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
        
        for label in ["induction", "retrieval", "sink"]:
            subset = agg[agg["label"] == label]
            if len(subset) > 0:
                print(f"\n{label.capitalize()} Heads Summary:")
                print(f"  Mean top Q layer: {subset['top_q_layer'].mean():.1f}")
                print(f"  Mean top K layer: {subset['top_k_layer'].mean():.1f}")
                print(f"  Embed K contrib %: {(subset['k_embed_contrib'] / (subset['total_score'] + 1e-6)).mean() * 100:.1f}%")

if __name__ == "__main__":
    run_attribution()
