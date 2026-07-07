import os
import torch
import random
import pandas as pd
import numpy as np
import math
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODELS = {
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B": "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B"
}

# The Universal Algorithm
def classify_head(embed_k_pct, q_layer, k_layer):
    if embed_k_pct > 0.10:
        return "sink"
    elif embed_k_pct > 0.01 and q_layer > k_layer:
        return "retrieval"
    elif embed_k_pct <= 0.01 and q_layer > k_layer:
        return "induction"
    else:
        return "local"

def build_universal_head_classes(model_key, n_layers, n_heads):
    path = f"outputs/phase1/component_attribution_all_{model_key}.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return {}
        
    df = pd.read_csv(path)
    classes = {}
    for _, row in df.iterrows():
        l, h = int(row['layer']), int(row['head'])
        embed_k_pct = row['k_embed_contrib'] / (row['total_score'] + 1e-6)
        classes[(l, h)] = classify_head(embed_k_pct, row['top_q_layer'], row['top_k_layer'])
        
    return classes

def get_eager_model(model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        device_map="cuda",
        attn_implementation="eager"
    )

def build_selective_mask(seq_len, n_layers, n_heads, head_classes):
    """
    Builds a custom 4D mask of shape (1, num_heads, seq_len, seq_len)
    Local: window of 30
    Sink: BOS (0:4) + local window of 4
    Induction/Retrieval: full causal
    """
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = torch.zeros((n_layers, n_heads, seq_len, seq_len))
    mask = mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    
    for l in range(n_layers):
        for h in range(n_heads):
            # Fallback to induction (full causal) to prevent blindly pruning unknown heads
            cls = head_classes.get((l, h), "induction")
            
            if cls == "local":
                window_mask = torch.ones(seq_len, seq_len)
                window_mask = torch.tril(window_mask) - torch.tril(window_mask, diagonal=-30)
                mask[l, h] = mask[l, h].masked_fill(window_mask == 0, float('-inf'))
                
            elif cls == "sink":
                sink_mask = torch.zeros(seq_len, seq_len)
                sink_mask[:, :4] = 1 
                local_win = torch.tril(torch.ones(seq_len, seq_len)) - torch.tril(torch.ones(seq_len, seq_len), diagonal=-4)
                sink_mask = (sink_mask + local_win) > 0
                mask[l, h] = mask[l, h].masked_fill(~sink_mask, float('-inf'))
                
    return mask

def evaluate_perplexity(model, tokenizer, num_samples=20, head_classes=None, mode="baseline"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50]
    texts = texts[:num_samples]
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    total_loss = 0.0
    total_tokens = 0
    
    for text in tqdm(texts, desc=f"Evaluating PPL ({mode})"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        seq_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            if mode == "baseline":
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
            else:
                hooks = []
                mask = build_selective_mask(seq_len, n_layers, n_heads, head_classes).cuda()
                
                def get_hook(layer_idx):
                    def pre_hook(module, args, kwargs):
                        layer_mask = mask[layer_idx].unsqueeze(0) # (1, n_heads, seq_len, seq_len)
                        kwargs["attention_mask"] = layer_mask
                        return args, kwargs
                    return pre_hook
                    
                for l in range(n_layers):
                    hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(get_hook(l), with_kwargs=True))
                    
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
                for h in hooks:
                    h.remove()
                    
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
        
    ppl = math.exp(total_loss / total_tokens)
    return ppl

def run_benchmarks():
    for model_key, model_id in MODELS.items():
        print(f"\nEvaluating {model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = get_eager_model(model_id)
        model.eval()
        
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        
        head_classes = build_universal_head_classes(model_key, n_layers, n_heads)
        if not head_classes:
            print("Missing attribution labels.")
            continue
            
        print("  Running Full Model Baseline...")
        baseline_ppl = evaluate_perplexity(model, tokenizer, mode="baseline")
        print(f"  Baseline Perplexity: {baseline_ppl:.2f}")
        
        print("  Running Universal Algorithm Selective Attention...")
        router_ppl = evaluate_perplexity(model, tokenizer, head_classes=head_classes, mode="router")
        print(f"  Router Perplexity: {router_ppl:.2f}")
        
        # Random Router: Shuffle the classes!
        print("  Running Random Router (Shuffled Classes)...")
        items = list(head_classes.items())
        keys = [k for k, v in items]
        vals = [v for k, v in items]
        random.shuffle(vals)
        random_classes = dict(zip(keys, vals))
        
        rand_ppl = evaluate_perplexity(model, tokenizer, head_classes=random_classes, mode="router")
        print(f"  Random Router Perplexity: {rand_ppl:.2f}")

if __name__ == "__main__":
    run_benchmarks()
