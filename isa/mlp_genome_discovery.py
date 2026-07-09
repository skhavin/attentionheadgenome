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
    parser.add_argument("--dataset", type=str, default="dataset_discovery_40.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    
    # Minimal compute / VRAM handling
    model_kwargs = {"device_map": DEVICE}
    if "gemma" in args.model_name.lower():
        # Requires bitsandbytes if running on 8GB VRAM
        try:
            model_kwargs["load_in_8bit"] = True
        except:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    fact_prompts = [item for item in dataset if item["task_type"] == "fact_recall"]
    print(f"Loaded {len(fact_prompts)} Fact Recall prompts for MLP Discovery.")

    n_layers = model.config.num_hidden_layers
    target_layers = list(range(int(n_layers * 0.85), n_layers)) # Top 15% of layers
    
    unembed = model.lm_head
    
    mlp_taxonomy = {
        "Boost-Correct": [],
        "Suppress-RunnerUp": [],
        "Neutral": []
    }

    for item_idx, item in enumerate(tqdm(fact_prompts, desc="MLP Discovery")):
        prompt = item["prompt"]
        target_str = item["target"]
        target_id = get_first_token_id(tokenizer, target_str)
        
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # We need a runner-up token. We run a clean pass to find the top-2 predictions.
        with torch.no_grad():
            clean_outputs = model(**tokens)
            final_logits = clean_outputs.logits[0, -1, :]
            top2_indices = torch.topk(final_logits, 2).indices.tolist()
            if top2_indices[0] == target_id:
                runner_up_id = top2_indices[1]
            else:
                runner_up_id = top2_indices[0] # Model is wrong, runner up is its top choice
        
        mlp_outputs = {}
        def get_hook(layer_idx):
            def hook(module, inp, out):
                # out is the output of down_proj (the final MLP output added to residual stream)
                hs = out[0] if isinstance(out, tuple) else out
                mlp_outputs[layer_idx] = hs[0, -1, :].detach().clone()
            return hook
            
        hooks = []
        for l_idx in target_layers:
            # Handle architecture differences
            if "gemma" in args.model_name.lower() or "llama" in args.model_name.lower():
                layer_mlp = model.model.layers[l_idx].mlp.down_proj
            else:
                layer_mlp = model.model.layers[l_idx].mlp.down_proj
            hooks.append(layer_mlp.register_forward_hook(get_hook(l_idx)))
            
        with torch.no_grad():
            _ = model(**tokens)
            
        for hook in hooks:
            hook.remove()
            
        # Analyze MLP taxonomy
        for l_idx in target_layers:
            v_mlp = mlp_outputs[l_idx]
            vocab_shift = unembed(v_mlp.to(model.dtype))
            
            target_shift = vocab_shift[target_id].item()
            runner_up_shift = vocab_shift[runner_up_id].item()
            
            # Simple heuristic for discovery: 
            # If target shift > 1.0, it's strongly boosting the correct answer
            # If runner_up shift < -1.0, it's strongly suppressing the distractor
            if target_shift > 1.0 and runner_up_shift < target_shift:
                mlp_taxonomy["Boost-Correct"].append(f"Prompt{item_idx}_L{l_idx}")
            elif runner_up_shift < -1.0 and target_shift > runner_up_shift:
                mlp_taxonomy["Suppress-RunnerUp"].append(f"Prompt{item_idx}_L{l_idx}")
            else:
                mlp_taxonomy["Neutral"].append(f"Prompt{item_idx}_L{l_idx}")
                
    print(f"\n--- MLP Genome Discovery Complete ({args.model_name}) ---")
    print(f"Total MLPs Analyzed: {len(fact_prompts) * len(target_layers)}")
    for category, items in mlp_taxonomy.items():
        print(f"{category}: {len(items)} MLPs")
        
    with open("mlp_taxonomy_discovery.json", "w") as f:
        json.dump(mlp_taxonomy, f, indent=2)

if __name__ == "__main__":
    main()
