"""
export_all_kv_geometry.py
─────────────────────────
Precomputes PCA projections of the KV cache for EVERY layer and EVERY head
across all 4 architectures for an interactive HTML viewer.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

os.environ["HF_HOME"] = r"d:\.cache\huggingface"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "geometry")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    "openai-community/gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B"
]

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "A very quick brown fox often jumps over a very lazy dog. "
    "Why does the quick brown fox jump over the lazy dog? "
    "Because the quick brown fox loves to jump over the lazy dog! "
    "Scientists observe that the quick brown fox and the lazy dog "
    "are frequently found together in typing exercises."
)

def get_token_category(idx, token_str):
    t = token_str.strip().lower()
    if idx == 0: return "First Token"
    elif t in ['.', ',', '!', '?', ';', ':', 'Ġ.', 'Ġ,', 'Ġ!', 'Ġ?']: return "Punctuation"
    elif t in ['the', 'a', 'is', 'of', 'and', 'to', 'in', 'it', 'for', 'with', 'on', 'as', 'by', 'at', 'are', 'that']: return "Stopwords"
    elif t in ['fox', 'dog']: return "Key Entities (fox/dog)"
    elif t in ['quick', 'brown', 'lazy', 'jumps', 'jump']: return "Attributes/Actions"
    else: return "Other Words"

def main():
    # Load canonical labels
    labels_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "canonical_labels.json")
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            canonical = json.load(f)["models"]
    else:
        canonical = {}

    all_data = {
        "prompt": PROMPT,
        "models": {}
    }

    for model_id in MODELS:
        print(f"\nProcessing {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
            model.eval()

            inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"][0].tolist()
            
            # Save tokens and categories
            tokens = [tokenizer.decode([idx]) for idx in input_ids]
            categories = [get_token_category(i, t) for i, t in enumerate(tokens)]
            
            model_short = model_id.split("/")[-1].replace("openai-community/", "")
            if model_short == "gpt2-medium": model_short = "GPT-2"
            elif model_short == "Qwen2.5-0.5B": model_short = "Qwen-0.5B"
            elif model_short == "Qwen2.5-1.5B": model_short = "Qwen-1.5B"
            
            canon_model = canonical.get(model_short, {"heads": {}})

            with torch.no_grad():
                outputs = model(**inputs, use_cache=True, output_attentions=False)
            
            past_kv = outputs.past_key_values
            
            num_layers = model.config.num_hidden_layers
            num_q_heads = model.config.num_attention_heads
            num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
            heads_per_kv = num_q_heads // num_kv_heads

            model_data = {
                "tokens": tokens,
                "categories": categories,
                "num_layers": num_layers,
                "num_heads": num_q_heads,
                "heads": {}
            }

            for layer in range(num_layers):
                for head in range(num_q_heads):
                    kv_head = head // heads_per_kv
                    # Extract K vectors
                    K = past_kv[layer][0][0, kv_head, :, :].cpu().numpy().astype(np.float32)
                    
                    pca = PCA(n_components=3, random_state=42)
                    K_3d = pca.fit_transform(K)
                    var_explained = float(sum(pca.explained_variance_ratio_) * 100)
                    
                    head_key = f"{layer}_{head}"
                    label = canon_model["heads"].get(head_key, {}).get("label", "unknown")
                    
                    model_data["heads"][head_key] = {
                        "x": [round(float(v), 4) for v in K_3d[:, 0]],
                        "y": [round(float(v), 4) for v in K_3d[:, 1]],
                        "z": [round(float(v), 4) for v in K_3d[:, 2]],
                        "var": round(var_explained, 1),
                        "label": label
                    }
                print(f"  Layer {layer+1}/{num_layers} done.")

            all_data["models"][model_short] = model_data
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error on {model_id}: {e}")

    out_file = os.path.join(OUT_DIR, "all_kv_geometry.json")
    with open(out_file, "w") as f:
        json.dump(all_data, f)
    print(f"\nSaved all PCA data to {out_file}")

if __name__ == "__main__":
    main()
