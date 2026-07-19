import json
import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import binomtest

def classify_fact_to_comparison(text):
    comparatives = ['yes', 'no', 'true', 'false', 'smaller', 'larger', 'taller', 'older', 'bigger', 'greater', 'less', 'better', 'worse', 'higher', 'lower']
    text_lower = text.lower()
    tokens = re.findall(r'\b\w+\b', text_lower)
    if any(c in tokens for c in comparatives):
        return 'B' 
    if len(tokens) > 0:
        return 'A' 
    return 'C' 

def mcnemar_exact(n12, n21):
    n = n12 + n21
    if n == 0: return 1.0
    k = min(n12, n21)
    return binomtest(k, n, p=0.5).pvalue

def test_model(model_name, hf_id, layer_range):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=========================================")
    print(f"Testing Model: {model_name}")
    print(f"=========================================")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    with open("../outputs/dataset/trajectory_validation.json", "r") as f:
        val_prompts = json.load(f)
        
    categories_all = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
    cat_indices = {c: [] for c in categories_all}
    for i, p in enumerate(val_prompts):
        cat_indices[p["task_type"]].append(i)

    raw_T = torch.load(f"../outputs/trajectories/{model_name}/val_raw_trajectories.pt", map_location="cpu")
    
    src_idx = cat_indices["fact_recall"]
    tgt_idx = cat_indices["comparison"]
    src_centroid = raw_T[src_idx].mean(dim=0).to(model.dtype)
    tgt_centroid = raw_T[tgt_idx].mean(dim=0).to(model.dtype)
    
    countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua", "Argentina", "Armenia", "Australia", "Austria",
        "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan"
    ]
    prompts = [f"The capital of {c} is\nAnswer:" for c in countries]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    c = 5.0
    torch.manual_seed(42)
    
    results = {}
    
    for layer in layer_range:
        v_steer = (tgt_centroid - src_centroid)[layer].to(model.device)
        v_rand = torch.randn_like(v_steer)
        v_rand = v_rand / torch.norm(v_rand) * torch.norm(v_steer)
        
        target_module = model.model.layers[layer]
        
        # Real Vector
        def hook_real(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h[:, -1, :] += c * v_steer
                return (h,) + output[1:]
            else:
                output[:, -1, :] += c * v_steer
                return output
                
        handle = target_module.register_forward_hook(hook_real)
        with torch.no_grad():
            gen_real = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        handle.remove()
        
        outputs_real = tokenizer.batch_decode(gen_real[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        buckets_real = [classify_fact_to_comparison(txt) for txt in outputs_real]
        b_rate_real = buckets_real.count('B') / len(buckets_real)
        
        # Random Vector
        def hook_rand(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h[:, -1, :] += c * v_rand
                return (h,) + output[1:]
            else:
                output[:, -1, :] += c * v_rand
                return output
                
        handle = target_module.register_forward_hook(hook_rand)
        with torch.no_grad():
            gen_rand = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        handle.remove()
        
        outputs_rand = tokenizer.batch_decode(gen_rand[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        buckets_rand = [classify_fact_to_comparison(txt) for txt in outputs_rand]
        b_rate_rand = buckets_rand.count('B') / len(buckets_rand)
        
        n12 = n21 = 0
        for r_real, r_rand in zip(buckets_real, buckets_rand):
            if r_real == 'B' and r_rand != 'B': n12 += 1
            elif r_real != 'B' and r_rand == 'B': n21 += 1
        p_val = mcnemar_exact(n12, n21)
        
        print(f"Layer {layer:02d} | Real B: {b_rate_real*100:5.1f}% | Rand B: {b_rate_rand*100:5.1f}% | p={p_val:.4f}")
        
        if b_rate_real >= 0.5 and p_val < 0.05:
            print(f"  >>> PASSED PRE-REGISTERED BAR (L{layer}) <<<")
    
    # Free up memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

def main():
    # Llama 3.2 1B (16 layers). Target window: ~25-55% -> layers 4 to 9.
    test_model("Llama-3.2-1B", "meta-llama/Llama-3.2-1B", range(4, 10))
    # Phi-1.5 (24 layers). Target window: ~25-55% -> layers 6 to 13.
    # Note: phi-1_5 model uses a different architecture name, but HF unifies it usually.
    # We will test phi-1_5 if Llama completes smoothly.

if __name__ == "__main__":
    main()
