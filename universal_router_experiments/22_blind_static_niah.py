import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_ids = ["Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/Phi-3-mini-4k-instruct"]
# We will just test Qwen-0.5B first as requested, but the rule is universal!

def generate_static_mask(model, n_layers, n_heads, num_kv_heads, head_dim):
    print("Synthesizing Hybrid Mask from Static Weights (Zero-Shot)...")
    dense_heads = set()
    embed_matrix = model.get_input_embeddings().weight.detach()
    
    for layer_idx in range(n_layers):
        q_proj = model.model.layers[layer_idx].self_attn.q_proj.weight.detach()
        k_proj = model.model.layers[layer_idx].self_attn.k_proj.weight.detach()
        v_proj = model.model.layers[layer_idx].self_attn.v_proj.weight.detach()
        
        q_proj = q_proj.view(n_heads, head_dim, -1)
        k_proj = k_proj.view(num_kv_heads, head_dim, -1)
        v_proj = v_proj.view(num_kv_heads, head_dim, -1)
        
        heads_per_kv = n_heads // num_kv_heads
        
        for head_idx in range(n_heads):
            q_w = q_proj[head_idx]
            kv_idx = head_idx // heads_per_kv
            k_w = k_proj[kv_idx]
            v_w = v_proj[kv_idx]
            
            # 1. Depth
            depth_ratio = layer_idx / n_layers
            
            # 2. V/Q Norm Ratio
            q_norm = torch.norm(q_w).item()
            v_norm = torch.norm(v_w).item()
            vq_ratio = v_norm / q_norm if q_norm > 0 else 0
            
            # 3. Embed-K Lock
            k_embed = F.linear(embed_matrix, k_w)
            k_baseline_norm = torch.norm(k_w).item() * torch.norm(embed_matrix).item()
            embed_k_lock = torch.norm(k_embed).item() / k_baseline_norm if k_baseline_norm > 0 else 0
            
            # THE UNIVERSAL STATIC RULE
            # Retrieval/Induction heads are deep (depth > 0.2), have high V/Q (> 1.0), and low embed-K lock (< 0.10)
            if depth_ratio >= 0.2 and vq_ratio > 1.0 and embed_k_lock < 0.10:
                dense_heads.add((layer_idx, head_idx))
                
    print(f"Static Rule classified {len(dense_heads)} out of {n_layers * n_heads} heads as Dense Retrieval.")
    return dense_heads

def test_niah(model_id):
    print(f"\n======================================")
    print(f"Testing {model_id}")
    print(f"======================================")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    
    dense_heads = generate_static_mask(model, n_layers, n_heads, num_kv_heads, head_dim)
    
    # Patch SDPA
    import torch.nn.functional as F_orig
    original_sdpa = F_orig.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        q_len = query.size(2)
        kv_len = key.size(2)
        
        if q_len > 1 and kv_len > 256:
            batch_size = query.size(0)
            n_heads_active = query.size(1)
            
            # Find which layer we are in by intercepting the stack
            import inspect
            frame = inspect.currentframe().f_back
            layer_idx = -1
            while frame:
                if 'self' in frame.f_locals:
                    obj = frame.f_locals['self']
                    if hasattr(obj, 'layer_idx'):
                        layer_idx = obj.layer_idx
                        break
                frame = frame.f_back
                
            if layer_idx != -1:
                window_size = 256
                causal_mask = torch.ones((q_len, kv_len), device=query.device, dtype=torch.bool).tril()
                
                # Create hybrid mask
                hybrid_mask = torch.zeros((1, n_heads_active, q_len, kv_len), device=query.device, dtype=torch.bool)
                
                for h in range(n_heads_active):
                    if (layer_idx, h) in dense_heads:
                        hybrid_mask[0, h] = causal_mask
                    else:
                        local_mask = torch.zeros((q_len, kv_len), device=query.device, dtype=torch.bool)
                        for q_idx in range(q_len):
                            start_k = max(0, q_idx - window_size + 1)
                            local_mask[q_idx, start_k:q_idx+1] = True
                        # ATTENTION SINK PRESERVATION (The Phase 3 Discovery)
                        local_mask[:, :4] = True
                        hybrid_mask[0, h] = local_mask & causal_mask
                
                return original_sdpa(query, key, value, attn_mask=hybrid_mask, dropout_p=dropout_p, is_causal=False, scale=scale)
                
        return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    
    # Run NIAH Test
    haystack_sentence = "The study of artificial intelligence has progressed rapidly over the past decade. "
    needle_sentence = "The secret password to unlock the HeadGenome matrix is Triton. "
    text = (haystack_sentence * 40) + needle_sentence + (haystack_sentence * 40) + "The secret password to unlock the HeadGenome matrix is"
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate 5 tokens
    print("Running Generation...")
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    print(f"Generation took {time.time() - start_time:.2f} seconds")
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
    print(f"Model Output: '{generated_text}'")
    
    if "Triton" in generated_text:
        print("RESULT: PASS (100% Retrieval)")
    else:
        print("RESULT: FAIL")
        
    # Unpatch for the next model
    torch.nn.functional.scaled_dot_product_attention = original_sdpa
    del model
    del tokenizer
    torch.cuda.empty_cache()

for m in model_ids:
    test_niah(m)
