import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_canonical_labels(model_name):
    label_map = {
        "Qwen/Qwen2.5-1.5B": "Qwen-1.5B",
        "unsloth/Llama-3.2-1B": "Llama-3.2-1B",
        "Qwen/Qwen2.5-0.5B": "Qwen-0.5B"
    }
    key = label_map.get(model_name, model_name)
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "canonical_labels.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("models", {}).get(key, {}).get("heads", {})

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

def get_first_token_id(tokenizer, text):
    """Tokenize the target text and return the first token id."""
    # We strip any weird BOS tokens if they are added
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(tokens) > 0:
        return tokens[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", type=str, default="dataset_60.json")
    parser.add_argument("--output", type=str, default="logit_lens_stats.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    
    canonical_dict = load_canonical_labels(args.model_name)
    
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        final_norm = model.model.norm
        n_heads = model.config.num_attention_heads
    else:
        raise ValueError("Unsupported model architecture.")
        
    lm_head = model.lm_head
    
    results = {
        "fact_recall": [],
        "pattern_induction": [],
        "niah": []
    }
    
    for item in dataset:
        prompt_id = item["id"]
        task_type = item["task_type"]
        prompt = item["prompt"]
        
        # Determine target token ID
        if task_type == "niah":
            target_str = item["target_full"]
        else:
            target_str = item["target"]
            
        target_token_id = get_first_token_id(tokenizer, target_str)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        hidden_states_per_layer = {}
        hooks = []
        def get_hook(layer_idx):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hidden_states_per_layer[layer_idx] = hs[0, -1, :].detach().clone()
            return hook
            
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(get_hook(i)))
            
        with torch.no_grad():
            _ = model(**inputs)
            
        for hook in hooks:
            hook.remove()
            
        shift_layer = -1
        
        for i in range(len(layers)):
            hs = hidden_states_per_layer[i]
            hs_normed = final_norm(hs)
            logits = lm_head(hs_normed)
            pred_id = torch.argmax(logits, dim=-1).item()
            
            if pred_id == target_token_id:
                shift_layer = i
                break
                
        # Final output check
        final_hs = hidden_states_per_layer[len(layers)-1]
        final_normed = final_norm(final_hs)
        final_logits = lm_head(final_normed)
        final_pred_id = torch.argmax(final_logits, dim=-1).item()
        
        success = (final_pred_id == target_token_id)
        
        # Calculate taxonomy at shift layer and layer before
        target_heads_present = False
        if shift_layer != -1:
            tax = get_layer_taxonomy(shift_layer, n_heads, canonical_dict)
            tax_prev = get_layer_taxonomy(max(0, shift_layer - 1), n_heads, canonical_dict)
            
            # For Fact Recall and NIAH, we care about Retrieval or Induction heads.
            # For Pattern Induction, we care about Induction heads.
            if task_type in ["fact_recall", "niah"]:
                if tax["retrieval"] > 0 or tax["induction"] > 0 or tax_prev["retrieval"] > 0 or tax_prev["induction"] > 0:
                    target_heads_present = True
            elif task_type == "pattern_induction":
                if tax["induction"] > 0 or tax_prev["induction"] > 0:
                    target_heads_present = True
        
        results[task_type].append({
            "id": prompt_id,
            "success": success,
            "shift_layer": shift_layer,
            "target_heads_present": target_heads_present
        })
        print(f"Processed {prompt_id} - Success: {success} - Shift: {shift_layer} - Heads: {target_heads_present}")
        
    # Compile Statistics
    stats = {}
    for task_type, items in results.items():
        total = len(items)
        successes = sum(1 for x in items if x["success"])
        failures = total - successes
        
        true_positives = sum(1 for x in items if x["success"] and x["target_heads_present"])
        false_positives = sum(1 for x in items if not x["success"] and x["target_heads_present"])
        false_negatives = sum(1 for x in items if x["success"] and not x["target_heads_present"])
        
        stats[task_type] = {
            "total": total,
            "success_rate": f"{successes}/{total}",
            "true_positives (Success + Head)": true_positives,
            "false_positives (Fail + Head)": false_positives,
            "false_negatives (Success + No Head)": false_negatives
        }
        
    with open(args.output, "w") as f:
        json.dump({"stats": stats, "raw": results}, f, indent=2)
        
    print(f"\n[DONE] Saved stats to {args.output}")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
