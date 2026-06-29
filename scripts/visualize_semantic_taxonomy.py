import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurations
MODELS = {
    "GPT-2": "gpt2-medium",
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B": "Qwen/Qwen2.5-1.5B"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs/phase9_semantics"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = {
    "Retrieval": "The capital of France is Paris. The capital of Japan is Tokyo. The capital of France is",
    "Induction": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
    "Local": "The large green building on the corner of the street is currently undergoing major renovations.",
    "Sink": "The large green building on the corner of the street is currently undergoing major renovations."
}

def load_best_heads():
    with open("outputs/canonical_labels.json", "r") as f:
        data = json.load(f)
    
    best_heads = {}
    for model_name, model_info in data["models"].items():
        if model_name not in MODELS:
            continue
            
        # Group heads by label
        heads_by_label = {"sink": [], "local": [], "retrieval": [], "induction": []}
        for hid, hinfo in model_info["heads"].items():
            heads_by_label[hinfo["label"]].append(hinfo)
        
        # Pick highest delta/entropy magnitude
        model_best = {}
        for label, heads in heads_by_label.items():
            if not heads:
                continue
            if label == "sink":
                best_head = min(heads, key=lambda x: x.get("match_entropy", 999))
            elif label == "retrieval":
                best_head = max(heads, key=lambda x: x.get("delta", -999))
            elif label == "induction":
                best_head = min(heads, key=lambda x: x.get("delta", 999))
            else: # local
                # For local, pick one with low delta near 0
                best_head = min(heads, key=lambda x: abs(x.get("delta", 999)))
                
            model_best[label.capitalize()] = best_head
        best_heads[model_name] = model_best
    return best_heads

def get_attention_weights(model, tokenizer, prompt, layer, head):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # attentions is a tuple of (batch, num_heads, seq_len, seq_len)
    attn_matrix = outputs.attentions[layer][0, head].cpu().numpy()
    
    # Get the attention weights for the last token attending to all previous tokens
    last_token_attn = attn_matrix[-1, :]
    
    # Get the tokens
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]
    
    return tokens, last_token_attn

def generate_html(all_data):
    html = """
    <html>
    <head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 40px; }
        h1 { text-align: center; color: #2c3e50; }
        h2 { border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-top: 40px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .cell { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .head-title { font-weight: bold; font-size: 1.1em; margin-bottom: 15px; color: #e74c3c; }
        .token { display: inline-block; padding: 2px 4px; margin: 2px; border-radius: 3px; border: 1px solid transparent; }
        .token-wrapper { line-height: 2.0; font-size: 14px; font-family: monospace; }
        .info { font-size: 0.9em; color: #7f8c8d; margin-bottom: 10px; }
    </style>
    </head>
    <body>
    <h1>HeadGenome: Semantic Universality Matrix</h1>
    <p style="text-align: center; max-width: 800px; margin: 0 auto; color: #555;">
        This visualization proves the semantic universality of the HeadGenome taxonomy. 
        Structurally categorized heads (Local, Retrieval, Induction, Sink) are shown side-by-side across architectures, 
        highlighting the exact tokens they attend to. The darker the red, the higher the attention mass.
    </p>
    """
    
    for task_name in ["Retrieval", "Induction", "Local", "Sink"]:
        html += f"<h2>{task_name} Task</h2><div class='grid'>"
        for model_name in ["GPT-2", "Qwen-0.5B", "Qwen-1.5B"]:
            html += f"<div class='cell'>"
            html += f"<h3>{model_name}</h3>"
            
            if model_name not in all_data or task_name not in all_data[model_name]:
                html += f"<p>No {task_name} head found for this model.</p></div>"
                continue
                
            task_data = all_data[model_name][task_name]
            layer = task_data['layer']
            head = task_data['head']
            delta = task_data.get('delta', 'N/A')
            
            html += f"<div class='head-title'>Layer {layer}, Head {head}</div>"
            html += f"<div class='info'>Classification Metric ($\\Delta$): {delta}</div>"
            html += "<div class='token-wrapper'>"
            
            tokens = task_data["tokens"]
            weights = task_data["weights"]
            
            import math
            clean_weights = [0 if math.isnan(w) else w for w in weights]
            
            # Normalize weights for visualization
            max_w = max(clean_weights) if len(clean_weights) > 0 else 1
            if max_w == 0: max_w = 1
            
            for t, w in zip(tokens, clean_weights):
                intensity = w / max_w
                # Color scale: white to dark red
                r = 255
                g = int(255 * (1 - intensity))
                b = int(255 * (1 - intensity))
                
                # Make text white if background is very dark
                text_color = "white" if intensity > 0.6 else "black"
                
                safe_t = str(t).replace("<", "&lt;").replace(">", "&gt;")
                
                html += f"<span class='token' style='background-color: rgb({r},{g},{b}); color: {text_color};' title='Weight: {w:.4f}'>{safe_t}</span>"
            
            html += "</div></div>"
        html += "</div>"
        
    html += "</body></html>"
    
    out_path = os.path.join(OUT_DIR, "universality_matrix.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Visualization saved to {out_path}")


def main():
    print("Loading canonical labels...")
    best_heads = load_best_heads()
    
    all_data = {}
    
    for model_name, hf_id in MODELS.items():
        if model_name not in best_heads:
            continue
            
        print(f"\nProcessing {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(hf_id, attn_implementation="eager", torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
        model.eval().to(DEVICE)
        
        all_data[model_name] = {}
        
        for task_name, head_info in best_heads[model_name].items():
            print(f"  Extracting {task_name} (L{head_info['layer']}H{head_info['head_idx']})")
            
            prompt = PROMPTS[task_name]
            layer = head_info["layer"]
            head = head_info["head_idx"]
            
            tokens, weights = get_attention_weights(model, tokenizer, prompt, layer, head)
            
            all_data[model_name][task_name] = {
                "layer": layer,
                "head": head,
                "delta": head_info.get("delta", "N/A"),
                "tokens": tokens,
                "weights": weights.tolist()
            }
            
        # Free memory
        del model
        torch.cuda.empty_cache()
        
    # Save raw json
    json_path = os.path.join(OUT_DIR, "semantic_attention_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nRaw data saved to {json_path}")
    
    # Generate HTML
    generate_html(all_data)


if __name__ == "__main__":
    main()
