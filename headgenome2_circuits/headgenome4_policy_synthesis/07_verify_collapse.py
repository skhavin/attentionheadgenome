import os
import torch
import pandas as pd
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-0.5B"

def get_eager_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float32, 
        device_map="cuda",
        attn_implementation="eager"
    )

def build_canonical_head_classes():
    with open("outputs/canonical_labels.json", "r") as f:
        data = json.load(f)
    heads = data["models"]["Qwen-0.5B"]["heads"]
    classes = {}
    for k, v in heads.items():
        classes[(v["layer"], v["head_idx"])] = v["label"]
    return classes

def classify_head(embed_k_pct, q_layer, k_layer):
    if embed_k_pct > 0.10:
        return "sink"
    elif embed_k_pct > 0.01 and q_layer > k_layer:
        return "retrieval"
    elif embed_k_pct <= 0.01 and q_layer > k_layer:
        return "induction"
    else:
        return "local"

def build_universal_head_classes():
    df = pd.read_csv("outputs/phase1/component_attribution_all_Qwen-0.5B.csv")
    classes = {}
    for _, row in df.iterrows():
        l, h = int(row['layer']), int(row['head'])
        embed_k_pct = row['k_embed_contrib'] / (row['total_score'] + 1e-6)
        classes[(l, h)] = classify_head(embed_k_pct, row['top_q_layer'], row['top_k_layer'])
    return classes

def build_selective_mask(seq_len, n_layers, n_heads, head_classes, window_size=30):
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = torch.zeros((n_layers, n_heads, seq_len, seq_len))
    mask = mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    
    for l in range(n_layers):
        for h in range(n_heads):
            cls = head_classes.get((l, h), "induction")
            
            if cls == "local":
                window_mask = torch.ones(seq_len, seq_len)
                window_mask = torch.tril(window_mask) - torch.tril(window_mask, diagonal=-window_size)
                mask[l, h] = mask[l, h].masked_fill(window_mask == 0, float('-inf'))
                
            elif cls == "sink":
                sink_mask = torch.zeros(seq_len, seq_len)
                sink_mask[:, :4] = 1 
                local_win = torch.tril(torch.ones(seq_len, seq_len)) - torch.tril(torch.ones(seq_len, seq_len), diagonal=-4)
                sink_mask = (sink_mask + local_win) > 0
                mask[l, h] = mask[l, h].masked_fill(~sink_mask, float('-inf'))
                
    return mask

def evaluate_perplexity(model, tokenizer, head_classes=None, window_size=30):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50][:20]
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    total_loss = 0.0
    total_tokens = 0
    
    for text in tqdm(texts):
        # Force a long enough sequence to ensure pruning actually happens (e.g. 1024)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        seq_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            if head_classes is None:
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
            else:
                hooks = []
                mask = build_selective_mask(seq_len, n_layers, n_heads, head_classes, window_size).cuda()
                
                def get_hook(layer_idx):
                    def pre_hook(module, args, kwargs):
                        kwargs["attention_mask"] = mask[layer_idx].unsqueeze(0)
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
        
    return math.exp(total_loss / total_tokens)

def main():
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = get_eager_model()
    model.eval()
    
    print("Evaluating Baseline...")
    base_ppl = evaluate_perplexity(model, tokenizer, head_classes=None)
    print(f"Baseline PPL: {base_ppl:.2f}")
    
    univ_classes = build_universal_head_classes()
    windows = [30, 128, 256, 512, 1024]
    
    for w in windows:
        print(f"\nEvaluating Universal Router (Local Window = {w})...")
        univ_ppl = evaluate_perplexity(model, tokenizer, head_classes=univ_classes, window_size=w)
        print(f"Universal Router (W={w}) PPL: {univ_ppl:.2f}")

if __name__ == "__main__":
    main()
