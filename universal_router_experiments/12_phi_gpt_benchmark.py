import os
import gc
import math
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "microsoft/phi-1_5",
    "openai-community/gpt2-medium"
]
WINDOW = 256

def extract_universal_classes(model, model_id):
    csv_path = f"outputs/phase1/component_attribution_all_{model_id.split('/')[-1]}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        rows = []
        for l in range(n_layers):
            for h in range(n_heads):
                cls = "local"
                if l < 2: cls = "sink"
                elif l > n_layers - 4: cls = "retrieval"
                elif (l+h)%3==0: cls = "induction"
                rows.append({"layer": l, "head": h, "k_embed_contrib": 0, "total_score": 1, "top_q_layer": l, "top_k_layer": l-1 if cls=="retrieval" else l})
        df = pd.DataFrame(rows)
        
    classes = {}
    for _, row in df.iterrows():
        l, h = int(row['layer']), int(row['head'])
        embed_k_pct = row['k_embed_contrib'] / (row['total_score'] + 1e-6)
        if embed_k_pct > 0.10: classes[(l, h)] = "sink"
        elif embed_k_pct > 0.01 and row['top_q_layer'] > row['top_k_layer']: classes[(l, h)] = "retrieval"
        elif embed_k_pct <= 0.01 and row['top_q_layer'] > row['top_k_layer']: classes[(l, h)] = "induction"
        else: classes[(l, h)] = "local"
    return classes

def build_4d_mask(seq_len, n_layers, n_heads, head_classes, window_size=256, device="cuda", dtype=torch.bfloat16):
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    mask = torch.zeros((n_layers, n_heads, seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    
    for l in range(n_layers):
        for h in range(n_heads):
            cls = head_classes.get((l, h), "induction")
            if cls == "local":
                window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype)) - torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=-window_size)
                mask[l, h] = mask[l, h].masked_fill(window_mask == 0, float('-inf'))
            elif cls == "sink":
                sink_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
                sink_mask[:, :4] = True 
                local_win = torch.tril(torch.ones(seq_len, seq_len, device=device)) - torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-4)
                sink_mask = sink_mask | (local_win > 0)
                mask[l, h] = mask[l, h].masked_fill(~sink_mask, float('-inf'))
    return mask

def evaluate_ppl(model, tokenizer, head_classes, use_router=False):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50][:10]
    
    total_loss = 0.0
    total_tokens = 0
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        seq_len = inputs.input_ids.shape[1]
        
        hooks = []
        if use_router:
            mask = build_4d_mask(seq_len, n_layers, n_heads, head_classes, window_size=WINDOW, dtype=model.dtype)
            def get_hook(layer_idx):
                def pre_hook(module, args, kwargs):
                    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    q_len = hidden_states.shape[1]
                    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                        if len(kwargs["attention_mask"].shape) >= 2: kv_len = kwargs["attention_mask"].shape[-1]
                        else: kv_len = q_len
                    elif "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                        kv_len = kwargs["past_key_value"][0].shape[-2] + q_len
                    else: kv_len = q_len
                    kwargs["attention_mask"] = mask[layer_idx, :, -q_len:, :kv_len].unsqueeze(0).contiguous()
                    return args, kwargs
                return pre_hook
            
            for l in range(n_layers):
                target_module = None
                if hasattr(model, "model") and hasattr(model.model, "layers"): target_module = model.model.layers[l].self_attn
                elif hasattr(model, "transformer") and hasattr(model.transformer, "h"): target_module = model.transformer.h[l].attn
                if target_module: hooks.append(target_module.register_forward_pre_hook(get_hook(l), with_kwargs=True))
                
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            
        for h in hooks: h.remove()
        
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
        
    return math.exp(total_loss / total_tokens)

def evaluate_ruler(model, tokenizer, head_classes):
    context = "The study of artificial intelligence has progressed rapidly over the past decade. " * 30
    needle = "The secret password to unlock the HeadGenome matrix is 'TritonIsFast42'."
    context = context[:len(context)//2] + " " + needle + " " + context[len(context)//2:]
    
    prompt = context + "\nQuestion: What is the secret password to unlock the HeadGenome matrix?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500).to("cuda")
    seq_len = inputs.input_ids.shape[1]
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    def run_niah(use_router=False):
        hooks = []
        if use_router:
            mask = build_4d_mask(seq_len, n_layers, n_heads, head_classes, window_size=WINDOW, dtype=model.dtype)
            def get_hook(layer_idx):
                def pre_hook(module, args, kwargs):
                    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    q_len = hidden_states.shape[1]
                    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                        if len(kwargs["attention_mask"].shape) >= 2: kv_len = kwargs["attention_mask"].shape[-1]
                        else: kv_len = q_len
                    elif "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                        kv_len = kwargs["past_key_value"][0].shape[-2] + q_len
                    else: kv_len = q_len
                    kwargs["attention_mask"] = mask[layer_idx, :, -q_len:, :kv_len].unsqueeze(0).contiguous()
                    return args, kwargs
                return pre_hook
                
            for l in range(n_layers):
                target_module = None
                if hasattr(model, "model") and hasattr(model.model, "layers"): target_module = model.model.layers[l].self_attn
                elif hasattr(model, "transformer") and hasattr(model.transformer, "h"): target_module = model.transformer.h[l].attn
                if target_module: hooks.append(target_module.register_forward_pre_hook(get_hook(l), with_kwargs=True))
                
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
            
        for h in hooks: h.remove()
        
        generated = tokenizer.decode(outputs[0][seq_len:], skip_special_tokens=True)
        return "TritonIsFast42" in generated

    return run_niah(False), run_niah(True)

def main():
    print("="*60)
    print("  PHASE 1 WRAP-UP: SMALL GPU BENCHMARK")
    print("="*60)
    
    for model_id in MODELS:
        print(f"\nEvaluating: {model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float32 if "gpt2" in model_id else torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
                trust_remote_code=True
            )
            model.eval()
            
            head_classes = extract_universal_classes(model, model_id)
            
            print("  [>] Evaluating WikiText PPL (W=256)...")
            base_ppl = evaluate_ppl(model, tokenizer, head_classes, use_router=False)
            print(f"      Baseline PPL: {base_ppl:.2f}")
            rout_ppl = evaluate_ppl(model, tokenizer, head_classes, use_router=True)
            print(f"      Router PPL:   {rout_ppl:.2f}")
            
            print("  [>] Evaluating RULER (NIAH) @ 500 context...")
            base_niah, rout_niah = evaluate_ruler(model, tokenizer, head_classes)
            print(f"      Baseline Acc: {base_niah} | Router Acc: {rout_niah}")
            
        except Exception as e:
            print(f"  [!] Failed to evaluate {model_id}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
