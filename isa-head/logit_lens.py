import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_canonical_labels(model_name):
    # Depending on model_name, we extract the label.
    # The canonical_labels.json maps model_names like "Qwen-1.5B"
    label_map = {
        "Qwen/Qwen2.5-1.5B": "Qwen-1.5B",
        "unsloth/Llama-3.2-1B": "Llama-3.2-1B",
        "gpt2-medium": "GPT-2-Medium",
        "Qwen/Qwen2.5-0.5B": "Qwen-0.5B"
    }
    key = label_map.get(model_name, model_name)
    
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "canonical_labels.json")
    if not os.path.exists(path):
        print(f"[WARN] No canonical_labels.json found at {path}")
        return {}
        
    with open(path, "r") as f:
        data = json.load(f)
        
    model_data = data.get("models", {}).get(key, {}).get("heads", {})
    return model_data

def get_layer_taxonomy(layer_idx, n_heads, canonical_dict):
    taxonomy = {"retrieval": 0, "induction": 0, "sink": 0, "local": 0, "unknown": 0}
    for h in range(n_heads):
        head_key = f"{layer_idx}_{h}"
        label = canonical_dict.get(head_key, {}).get("label", "unknown")
        if label in taxonomy:
            taxonomy[label] += 1
        else:
            taxonomy["unknown"] += 1
    return taxonomy

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="HF model name")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset.json")
    parser.add_argument("--output", type=str, default="logit_lens_results.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    
    canonical_dict = load_canonical_labels(args.model_name)
    
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
        
    results = {}
    
    # Identify the base model layers and norm depending on architecture
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        final_norm = model.model.norm
        n_heads = model.config.num_attention_heads
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        final_norm = model.transformer.ln_f
        n_heads = model.config.n_head
    else:
        raise ValueError("Unsupported model architecture for logit lens.")
        
    lm_head = model.lm_head
    
    for item in dataset:
        prompt_id = item["id"]
        prompt = item["prompt"]
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_id}")
        print(f"{prompt}")
        print(f"{'='*60}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # We will use hooks to capture hidden states
        hidden_states_per_layer = {}
        hooks = []
        
        def get_hook(layer_idx):
            def hook(module, inp, out):
                # out is usually a tuple (hidden_states, ...)
                hs = out[0] if isinstance(out, tuple) else out
                # Capture the last token's hidden state
                hidden_states_per_layer[layer_idx] = hs[0, -1, :].detach().clone()
            return hook
            
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(get_hook(i)))
            
        with torch.no_grad():
            outputs = model(**inputs)
            
        for hook in hooks:
            hook.remove()
            
        layer_logs = []
        
        # Now apply the lens
        for i in range(len(layers)):
            hs = hidden_states_per_layer[i]
            # Apply final layer norm
            hs_normed = final_norm(hs)
            # Apply lm_head
            logits = lm_head(hs_normed)
            
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)
            
            top_tokens = []
            for prob, idx in zip(top_probs, top_indices):
                token_str = tokenizer.decode([idx.item()])
                # Clean up newlines for display
                token_str = token_str.replace('\n', '\\n')
                top_tokens.append({"token": token_str, "prob": float(prob.item())})
                
            taxonomy = get_layer_taxonomy(i, n_heads, canonical_dict)
            tax_str = ", ".join([f"{v} {k.capitalize()}" for k, v in taxonomy.items() if v > 0])
            
            log_entry = {
                "layer": i,
                "top_predictions": top_tokens,
                "taxonomy": taxonomy
            }
            layer_logs.append(log_entry)
            
            # Print beautiful output
            top_3_str = ", ".join([f"'{t['token']}' ({t['prob']:.2f})" for t in top_tokens[:3]])
            print(f"Layer {i:<2}: [{top_3_str}] | Heads: {tax_str}")
            
        # Also log the final true prediction
        final_logits = outputs.logits[0, -1, :]
        final_probs = F.softmax(final_logits, dim=-1)
        final_prob, final_idx = torch.topk(final_probs, 1)
        final_token = tokenizer.decode([final_idx.item()])
        print(f"\nFinal Generated Token: '{final_token}'")
        
        results[prompt_id] = {
            "prompt": prompt,
            "target": item.get("target"),
            "final_generated_token": final_token,
            "layer_logs": layer_logs
        }
        
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[DONE] Saved results to {args.output}")

if __name__ == "__main__":
    main()
