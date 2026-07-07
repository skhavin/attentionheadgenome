import os
import time
import math
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen2.5-0.5B"
WINDOW = 256
SINK_SIZE = 4

def extract_universal_classes(model, tokenizer):
    csv_path = f"outputs/phase1/component_attribution_all_{MODEL_ID.split('/')[-1]}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Dummy fallback
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

def make_mask_mod(layer_roles, num_heads):
    """Creates a mask_mod for PyTorch flex_attention compiler."""
    critical_heads = [h for h, role in layer_roles.items() if role in ["retrieval", "induction"]]
    sink_heads = [h for h, role in layer_roles.items() if role == "sink"]
    
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_sink_tok = kv_idx < SINK_SIZE
        is_local = (q_idx - kv_idx) <= WINDOW
        
        is_critical = torch.zeros_like(causal, dtype=torch.bool)
        for ch in critical_heads:
            is_critical = is_critical | (h == ch)
            
        is_sink_head = torch.zeros_like(causal, dtype=torch.bool)
        for sh in sink_heads:
            is_sink_head = is_sink_head | (h == sh)
            
        full_mask = causal
        local_mask = causal & (is_sink_tok | is_local)
        
        return torch.where(is_critical, full_mask, local_mask)
        
    return mask_mod

def build_flex_monkeypatch(model, head_classes):
    print("\n[+] Compiling Production-Grade FlexAttention Triton Kernels...")
    n_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    layer_mask_mods = {}
    for l in range(n_layers):
        layer_roles = {h: head_classes.get((l, h), "local") for h in range(num_heads)}
        layer_mask_mods[l] = make_mask_mod(layer_roles, num_heads)
        
    original_forwards = {}
    
    def get_flex_forward(layer_idx, original_forward):
        def custom_forward(
            hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs
        ):
            module = model.model.layers[layer_idx].self_attn
            bsz, q_len, _ = hidden_states.size()

            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)

            num_heads_val = model.config.num_attention_heads
            num_kv_heads_val = model.config.num_key_value_heads
            
            query_states = query_states.view(bsz, q_len, num_heads_val, module.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads_val, module.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads_val, module.head_dim).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = module.rotary_emb(value_states, position_ids)
                
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_position)
                
            kv_len = key_states.shape[2]
            
            mask_mod = layer_mask_mods[layer_idx]
            
            # create_block_mask compiles the python mask_mod into a Triton kernel
            block_mask = create_block_mask(
                mask_mod,
                B=None, H=num_heads_val, Q_LEN=q_len, KV_LEN=kv_len,
                device=query_states.device,
                _compile=True
            )
            
            scale = 1.0 / math.sqrt(module.head_dim)
            
            # flex_attention automatically executes a highly optimized block-sparse Triton kernel
            attn_output = flex_attention(query_states, key_states, value_states, block_mask=block_mask, scale=scale)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = module.o_proj(attn_output)

            return attn_output, None, past_key_value
            
        return custom_forward

    for l in range(n_layers):
        layer = model.model.layers[l]
        original_forwards[l] = layer.self_attn.forward
        layer.self_attn.forward = get_flex_forward(l, original_forwards[l])
        
    return original_forwards


def measure_speedup(model, tokenizer, head_classes):
    print("\n[>] Measuring Production-Grade TTFT Speedup (4096 tokens)...")
    prompt = "The quick brown fox jumps over the lazy dog. " * 500
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    
    for _ in range(2):
        with torch.no_grad():
            model(**inputs)
            
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()
    base_time = time.perf_counter() - t0
    print(f"      Baseline TTFT: {base_time*1000:.1f}ms")
    
    orig_fwds = build_flex_monkeypatch(model, head_classes)
    
    print("      (Compiling Triton kernels... this takes a few seconds)")
    for _ in range(2):
        with torch.no_grad():
            model(**inputs)
            
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()
    flex_time = time.perf_counter() - t0
    
    print(f"      FlexAttention TTFT: {flex_time*1000:.1f}ms")
    print(f"      PROD-GRADE SPEEDUP: {base_time / flex_time:.2f}x")
    
    for l, orig_fwd in orig_fwds.items():
        model.model.layers[l].self_attn.forward = orig_fwd

def main():
    print("="*60)
    print("  PRODUCTION-GRADE FLEX ATTENTION SPEEDUP")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    model.eval()
    
    head_classes = extract_universal_classes(model, tokenizer)
    measure_speedup(model, tokenizer, head_classes)

if __name__ == "__main__":
    main()
