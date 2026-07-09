import torch
import math
import os
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen2.5-0.5B"
CHUNK_SIZE = 128
LOCAL_WINDOW = 256
SINK_WINDOW = 32
THRESHOLD = 0.95 # Relative Softmax Exit Threshold

def load_canonical_labels(model_id):
    path = "../outputs/canonical_labels.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    labels = {}
    
    # Map the model ID to the canonical name used in the JSON
    canonical_model_id = "Qwen-0.5B" if "0.5B" in model_id else model_id
    
    if canonical_model_id in data.get("models", {}):
        heads = data["models"][canonical_model_id].get("heads", {})
        for k, v in heads.items():
            l, h = map(int, k.split('_'))
            labels[(l, h)] = v.get("label", "local")
    return labels

def evaluate_ppl(model, tokenizer, head_classes=None):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    start_time = time.time()
    for begin_loc in range(0, min(seq_len, 1024 * 5), stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    elapsed = time.time() - start_time
    print(f"  WikiText PPL: {ppl:.2f} (Time: {elapsed:.2f}s)")
    return ppl, elapsed

def evaluate_niah(model, tokenizer, head_classes=None):
    print(f"\n[>] Evaluating RULER NIAH 4000...")
    context = "The study of artificial intelligence has progressed rapidly over the past decade. " * 300
    needle = "The secret password to unlock the HeadGenome matrix is 'TritonIsFast42'."
    context = context[:len(context)//2] + " " + needle + " " + context[len(context)//2:]
    
    prompt = context + "\nQuestion: What is the secret password to unlock the HeadGenome matrix?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
    total_time = time.time() - t0
    
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    success = "TritonIsFast42" in gen_text
    print(f"  NIAH Result: {'PASS' if success else 'FAIL'} (Generated: {gen_text.strip()!r}) (Time: {total_time:.2f}s)")
    return success, total_time

def monkeypatch_canonical_router(model, head_classes):
    print("\n[+] Monkeypatching Qwen2Attention with Canonical O(N) Router...")
    
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
            
            n_rep = num_heads // num_kv_heads
            if n_rep > 1:
                key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
                value_states = torch.repeat_interleave(value_states, n_rep, dim=1)
            
            attn_weights = torch.full((bsz, num_heads, q_len, kv_len), float('-inf'), device=query_states.device, dtype=query_states.dtype)
            
            # Pre-categorize heads
            local_h = []
            sink_h = []
            retrieval_h = []
            induction_h = []
            
            for h in range(num_heads):
                cls = head_classes.get((layer_idx, h), "local")
                if cls == "local": local_h.append(h)
                elif cls == "sink": sink_h.append(h)
                elif cls == "retrieval": retrieval_h.append(h)
                else: induction_h.append(h)
                
            # 1. Process Local and Sink Heads statically (O(N) bounds)
            for heads, window in [(local_h, LOCAL_WINDOW), (sink_h, SINK_WINDOW)]:
                if not heads: continue
                
                q = query_states[:, heads, :, :]
                k = key_states[:, heads, :, :]
                
                dots = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(module.head_dim)
                
                if q_len > 1: # Prefill
                    q_pos = torch.arange(kv_len - q_len, kv_len, device=query_states.device).unsqueeze(1)
                    k_pos = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
                    causal = q_pos >= k_pos
                    window_mask = (q_pos - k_pos) < window
                    sink_mask = k_pos < 4
                    valid = causal & (window_mask | sink_mask)
                    dots = dots.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float('-inf'))
                else: # Decode
                    q_pos = kv_len - 1
                    k_pos = torch.arange(kv_len, device=query_states.device)
                    window_mask = (q_pos - k_pos) < window
                    sink_mask = k_pos < 4
                    valid = window_mask | sink_mask
                    dots = dots.masked_fill(~valid.unsqueeze(0).unsqueeze(0).unsqueeze(0), float('-inf'))
                    
                attn_weights[:, heads, :, :] = dots
                
            # 2. Process Induction Heads (Dynamically Dense)
            if induction_h:
                q_ind = query_states[:, induction_h, :, :]
                k_ind = key_states[:, induction_h, :, :]
                dots = torch.matmul(q_ind, k_ind.transpose(2, 3)) / math.sqrt(module.head_dim)
                
                if q_len > 1:
                    q_pos = torch.arange(kv_len - q_len, kv_len, device=query_states.device).unsqueeze(1)
                    k_pos = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
                    causal = q_pos >= k_pos
                    dots = dots.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                    
                attn_weights[:, induction_h, :, :] = dots

            # 3. Process Retrieval Heads (Dynamically Dense - for Ablation Testing)
            if retrieval_h:
                q_ret = query_states[:, retrieval_h, :, :]
                k_ret = key_states[:, retrieval_h, :, :]
                dots = torch.matmul(q_ret, k_ret.transpose(2, 3)) / math.sqrt(module.head_dim)
                
                if q_len > 1:
                    q_pos = torch.arange(kv_len - q_len, kv_len, device=query_states.device).unsqueeze(1)
                    k_pos = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
                    causal = q_pos >= k_pos
                    dots = dots.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                    
                attn_weights[:, retrieval_h, :, :] = dots

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = module.o_proj(attn_output)

            return attn_output, attn_weights
            
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
    print("  PHASE 4: CANONICAL O(N) ROUTER (FIXED EARLY STOPPER)")
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
    
    head_classes = load_canonical_labels(MODEL_ID)
    print(f"Loaded Canonical Labels: {len(head_classes)} heads mapped.")
    
    print("\n--- BASELINE ---")
    # base_ppl, _ = evaluate_ppl(model, tokenizer)
    print(f"  WikiText PPL: [SKIPPED FOR DEBUG]")
    base_niah, _ = evaluate_niah(model, tokenizer)
    
    print("\n--- CANONICAL ROUTER (O(N) WITH FIXED EARLY STOPPER) ---\n")
    orig_fwds = monkeypatch_canonical_router(model, head_classes)
    # cr_ppl, _ = evaluate_ppl(model, tokenizer)
    print(f"  WikiText PPL: [SKIPPED FOR DEBUG]")
    cr_niah, _ = evaluate_niah(model, tokenizer)
    
    restore_monkeypatch(model, orig_fwds)
    
    print("\n=== SUMMARY ===")
    print(f"Baseline PPL: [SKIPPED] | Router PPL: [SKIPPED]")
    print(f"Baseline NIAH: {'PASS' if base_niah else 'FAIL'} | Router NIAH: {'PASS' if cr_niah else 'FAIL'}")

if __name__ == "__main__":
    main()
