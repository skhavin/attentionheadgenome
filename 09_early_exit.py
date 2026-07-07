import os
import sys
import math
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen2.5-0.5B"
THRESHOLD = 15.0  # Softmax threshold
WINDOW = 256
CHUNK_SIZE = 128

def extract_universal_classes(model, tokenizer):
    csv_path = f"outputs/phase1/component_attribution_all_{MODEL_ID.split('/')[-1]}.csv"
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

def evaluate_ppl(model, tokenizer, use_early_exit=False, head_classes=None):
    from datasets import load_dataset
    print(f"\n[>] Evaluating WikiText PPL (Early Exit={use_early_exit})...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50][:10]
    
    total_loss = 0.0
    total_tokens = 0
    total_time = 0.0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        seq_len = inputs.input_ids.shape[1]
        
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
        
    ppl = math.exp(total_loss / total_tokens)
    print(f"    PPL: {ppl:.2f} | Time: {total_time:.2f}s")
    return ppl, total_time

def evaluate_niah(model, tokenizer, use_early_exit=False, head_classes=None):
    print(f"\n[>] Evaluating RULER NIAH 4000 (Early Exit={use_early_exit})...")
    context = "The study of artificial intelligence has progressed rapidly over the past decade. " * 300
    needle = "The secret password to unlock the HeadGenome matrix is 'TritonIsFast42'."
    context = context[:len(context)//2] + " " + needle + " " + context[len(context)//2:]
    
    prompt = context + "\nQuestion: What is the secret password to unlock the HeadGenome matrix?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
    seq_len = inputs.input_ids.shape[1]
    
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
        
    generated = tokenizer.decode(outputs[0][seq_len:], skip_special_tokens=True)
    success = "TritonIsFast42" in generated
    print(f"    NIAH Success: {success} | Decode Time: {total_time:.2f}s | Output: {generated.strip()}")
    return success, total_time

def monkeypatch_early_exit(model, head_classes):
    print("\n[+] Monkeypatching Qwen2Attention with Chunked Early-Exit...")
    
    original_forwards = {}
    
    def get_custom_forward(layer_idx, original_forward):
        def custom_forward(
            hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs
        ):
            module = model.model.layers[layer_idx].self_attn
            bsz, q_len, _ = hidden_states.size()

            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)

            num_heads = model.config.num_attention_heads
            num_kv_heads = model.config.num_key_value_heads
            head_dim = module.head_dim
            
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = module.rotary_emb(value_states, position_ids)
                
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_position)
                
            kv_len = key_states.shape[2]
            
            # GQA Fix: Repeat KV heads to match Query heads before slicing
            n_rep = num_heads // num_kv_heads
            if n_rep > 1:
                key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
                value_states = torch.repeat_interleave(value_states, n_rep, dim=1)
            
            attn_weights = torch.full((bsz, num_heads, q_len, kv_len), float('-inf'), device=query_states.device, dtype=query_states.dtype)
            
            # To get ACTUAL speedup in PyTorch, we loop backwards and break when thresholds are met
            # If all heads have exited, we break the loop!
            # Since indexing specific queries/heads is slow in torch, we just track active_heads 
            active_heads = torch.ones(num_heads, dtype=torch.bool, device=query_states.device)
            
            for i in range(kv_len, 0, -CHUNK_SIZE):
                if not active_heads.any():
                    break
                    
                start = max(0, i - CHUNK_SIZE)
                end = i
                
                # Only compute matmul for active heads to save actual FLOPs
                active_idx = active_heads.nonzero(as_tuple=True)[0]
                q_active = query_states[:, active_idx, :, :]
                k_chunk_active = key_states[:, active_idx, start:end, :]
                
                q_dot_k = torch.matmul(q_active, k_chunk_active.transpose(2, 3)) / math.sqrt(module.head_dim)
                
                # Apply causal mask
                if q_len > 1:
                    q_pos = torch.arange(kv_len - q_len, kv_len, device=query_states.device).unsqueeze(1)
                    k_pos = torch.arange(start, end, device=query_states.device).unsqueeze(0)
                    causal = q_pos >= k_pos
                    q_dot_k = q_dot_k.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Scatter back to the main tensor
                attn_weights[:, active_idx, :, start:end] = q_dot_k
                
                # Check for Early Exit
                max_scores, _ = q_dot_k.max(dim=-1)
                
                # If a query in a head reached THRESHOLD, that head is done for that query.
                # To simplify vectorization for actual speedup, if ALL queries in a head reached threshold, we deactivate the head.
                # In decode mode (q_len=1), this works perfectly. In prefill, this is a conservative early exit.
                head_exit = (max_scores > THRESHOLD).all(dim=-1) # (bsz, len(active_idx))
                
                # We assume batch size 1 for simplicity of this speedup demo
                if bsz == 1:
                    exited_active_idx = head_exit[0]
                    active_heads[active_idx[exited_active_idx]] = False
                    
            # Post-process: Mask out older tokens for Local heads
            for h in range(num_heads):
                cls = head_classes.get((layer_idx, h), "local")
                if cls == "local" and q_len > 1:
                    local_mask = torch.tril(torch.ones(q_len, kv_len, device=query_states.device)) - torch.tril(torch.ones(q_len, kv_len, device=query_states.device), diagonal=-WINDOW)
                    attn_weights[:, h] = attn_weights[:, h].masked_fill(local_mask == 0, float('-inf'))
                    
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = module.o_proj(attn_output)

            return attn_output, None, past_key_value
            
        return custom_forward

    n_layers = model.config.num_hidden_layers
    for l in range(n_layers):
        layer = model.model.layers[l]
        original_forwards[l] = layer.self_attn.forward
        layer.self_attn.forward = get_custom_forward(l, original_forwards[l])
        
    return original_forwards

def restore_monkeypatch(model, original_forwards):
    for l, orig_fwd in original_forwards.items():
        model.model.layers[l].self_attn.forward = orig_fwd

def main():
    print("="*60)
    print("  PHASE 2: EARLY EXIT SOFTMAX EXPERIMENT")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager"
    )
    model.eval()
    
    head_classes = extract_universal_classes(model, tokenizer)
    
    print("\n--- BASELINE ---")
    base_ppl, base_time = evaluate_ppl(model, tokenizer)
    base_niah, base_niah_time = evaluate_niah(model, tokenizer)
    
    print("\n--- EARLY EXIT CHUNKED ATTENTION ---")
    orig_fwds = monkeypatch_early_exit(model, head_classes)
    ee_ppl, ee_time = evaluate_ppl(model, tokenizer, use_early_exit=True, head_classes=head_classes)
    ee_niah, ee_niah_time = evaluate_niah(model, tokenizer, use_early_exit=True, head_classes=head_classes)
    
    restore_monkeypatch(model, orig_fwds)

if __name__ == "__main__":
    main()
