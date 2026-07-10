import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_first_token_id(tokenizer, text):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokens[0] if tokens else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", type=str, default="dataset_60.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    fact_prompts = [item for item in dataset if item["task_type"] == "fact_recall"]
    print(f"Loaded {len(fact_prompts)} Fact Recall prompts for DLA verification.")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    # We will check the final 3 layers (where factual shift typically occurs)
    target_layers = [n_layers - 3, n_layers - 2, n_layers - 1]
    
    total_heads_checked = 0
    direct_factual_outputs = 0

    unembed = model.lm_head

    for item in tqdm(fact_prompts, desc="Running DLA"):
        prompt = item["prompt"]
        target_str = item["target"]
        target_token_id = get_first_token_id(tokenizer, target_str)
        
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        for layer_idx in target_layers:
            layer = model.model.layers[layer_idx]
            
            head_outputs = {}
            def get_hook(layer_id):
                def hook(module, inp, out):
                    v_concat = inp[0][0, -1, :] 
                    for h in range(n_heads):
                        v_isolated = torch.zeros_like(v_concat)
                        v_isolated[h * head_dim : (h+1) * head_dim] = v_concat[h * head_dim : (h+1) * head_dim]
                        head_update = F.linear(v_isolated, module.weight, module.bias)
                        head_outputs[f"{layer_id}_{h}"] = head_update
                return hook

            hook_handle = layer.self_attn.o_proj.register_forward_hook(get_hook(layer_idx))
            
            with torch.no_grad():
                _ = model(**tokens)
                
            hook_handle.remove()
            
            for h in range(n_heads):
                total_heads_checked += 1
                update = head_outputs[f"{layer_idx}_{h}"]
                
                # Project through unembed
                logits = unembed(update.to(model.dtype))
                top_indices = torch.topk(logits, 5).indices.tolist()
                
                if target_token_id in top_indices:
                    direct_factual_outputs += 1
                    print(f"\n[ALERT] Layer {layer_idx}, Head {h} directly output the target token '{target_str}' for prompt: '{prompt}'")

    print(f"\n--- DLA Statistical Verification Complete ---")
    print(f"Total Prompts: {len(fact_prompts)}")
    print(f"Total Attention Heads Checked: {total_heads_checked}")
    print(f"Direct Factual Outputs (Target in Top-5 Logits): {direct_factual_outputs}")
    
    if direct_factual_outputs == 0:
        print("\n[CONCLUSION] SUCCESS: 0% of Attention Heads directly output the factual target. The Intersection Circuit is statistically robust across N=20.")
    else:
        rate = (direct_factual_outputs / total_heads_checked) * 100
        print(f"\n[CONCLUSION] FAILED: {rate:.2f}% of heads directly output the target.")

if __name__ == "__main__":
    main()
