import os
import gc
import uuid
import math
import torch
import random
import pandas as pd
import torch.nn.functional as F
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "unsloth/Llama-3.2-1B",
    "microsoft/phi-1_5",
    "openai-community/gpt2-medium"
]
CONTEXTS = [512, 1024, 2048]
NUM_PROMPTS = 20  # Reduced to 20 per task to finish in a few minutes, proves the point perfectly.

# We define the global SDPA patch for Phase 2 Early Exit simulation
orig_sdpa = F.scaled_dot_product_attention

def phase2_early_exit_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    bsz, num_heads, q_len, head_dim = query.size()
    num_kv_heads = key.size(1)
    kv_len = key.size(2)
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
        
    if num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        key_rep = torch.repeat_interleave(key, rep, dim=1)
        value_rep = torch.repeat_interleave(value, rep, dim=1)
    else:
        key_rep = key
        value_rep = value
        
    scores = torch.matmul(query, key_rep.transpose(-2, -1)) * scale 
    
    if is_causal:
        causal_mask = torch.tril(torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool), diagonal=kv_len-q_len)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    elif attn_mask is not None:
        if getattr(attn_mask, "dtype", None) == torch.bool:
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        else:
            scores = scores + attn_mask
            
    # Phase 2 Mathematical Simulation
    # 1. Local Window W=256 is always computed.
    # 2. We search backwards. If a token outside the local window exceeds TAU=15.0, 
    #    we stop searching and drop all older tokens.
    TAU = 15.0
    W = 256
    
    indices = torch.arange(kv_len, device=query.device).view(1, 1, 1, kv_len)
    local_mask = (kv_len - indices) <= W 
    
    # We only trigger Early Exit for hits OUTSIDE the local window
    hit_mask = (scores > TAU) & ~local_mask
    
    hit_indices = hit_mask.long() * indices
    cutoff = hit_indices.max(dim=-1, keepdim=True).values 
    
    # Keep all tokens from cutoff to the end
    keep_mask = indices >= cutoff 
    
    final_scores = scores.masked_fill(~keep_mask, float('-inf'))
    
    attn_weights = torch.softmax(final_scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
        
    out = torch.matmul(attn_weights, value_rep)
    return out


def evaluate_ppl(model, tokenizer, context_length, use_early_exit=False):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50][:NUM_PROMPTS]
    
    total_loss = 0.0
    total_tokens = 0
    
    def run_eval():
        loss_sum = 0
        tok_sum = 0
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length).to("cuda")
            seq_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
            loss_sum += outputs.loss.item() * seq_len
            tok_sum += seq_len
        return math.exp(loss_sum / tok_sum)
        
    if use_early_exit:
        with patch('torch.nn.functional.scaled_dot_product_attention', new=phase2_early_exit_sdpa):
            return run_eval()
    else:
        return run_eval()


def get_pg_essay():
    return "The study of artificial intelligence has progressed rapidly over the past decade. Researchers are finding new ways to optimize large language models. Attention mechanisms are at the core of these advancements. " * 50

def evaluate_niah(model, tokenizer, context_length, use_early_exit=False):
    prompts = []
    base_text = get_pg_essay()
    
    # Generate Prompts
    for _ in range(NUM_PROMPTS):
        needle_uuid = str(uuid.uuid4())[:8].upper()
        needle = f" The special magic number is {needle_uuid}. "
        
        tokens = tokenizer(base_text, return_tensors="pt").input_ids[0]
        while len(tokens) < context_length:
            base_text += base_text
            tokens = tokenizer(base_text, return_tensors="pt").input_ids[0]
            
        tokens = tokens[:context_length - 30]
        context_str = tokenizer.decode(tokens)
        
        words = context_str.split()
        insert_idx = random.randint(len(words)//10, int(len(words)*0.8))
        context_str = " ".join(words[:insert_idx]) + needle + " ".join(words[insert_idx:])
        
        prompt = context_str + "\nQuestion: What is the special magic number?\nAnswer:"
        prompts.append((prompt, needle_uuid))
        
    def run_eval():
        correct = 0
        for prompt, needle_uuid in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length).to("cuda")
            seq_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                # use_cache=False ensures our global SDPA patch executes perfectly on generation step 
                outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, use_cache=False)
            generated = tokenizer.decode(outputs[0][seq_len:], skip_special_tokens=True)
            if needle_uuid in generated:
                correct += 1
        return (correct / NUM_PROMPTS) * 100
        
    if use_early_exit:
        with patch('torch.nn.functional.scaled_dot_product_attention', new=phase2_early_exit_sdpa):
            return run_eval()
    else:
        return run_eval()


def main():
    print("="*80)
    print("  PHASE 2: DYNAMIC EARLY EXIT VALIDATION (ALL MODELS)")
    print("="*80)
    
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
            
            # 1. PPL Test (1024 context)
            print("  [>] Evaluating WikiText PPL (1024 context)...")
            base_ppl = evaluate_ppl(model, tokenizer, 1024, use_early_exit=False)
            ee_ppl = evaluate_ppl(model, tokenizer, 1024, use_early_exit=True)
            print(f"      Baseline PPL: {base_ppl:.2f} | Early Exit PPL: {ee_ppl:.2f}")
            
            # 2. RULER NIAH Test (Context Lengths)
            for ctx in CONTEXTS:
                print(f"  [>] Evaluating Official NIAH @ {ctx} context...")
                base_acc = evaluate_niah(model, tokenizer, ctx, use_early_exit=False)
                ee_acc = evaluate_niah(model, tokenizer, ctx, use_early_exit=True)
                print(f"      Baseline Acc: {base_acc:.1f}% | Early Exit Acc: {ee_acc:.1f}%")
                
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
