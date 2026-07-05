"""
step18_routing_engine.py
-------------------------
Monkey-patches Qwen2.5-0.5B's attention forward() with a per-head
dispatch table that routes each head to its pre-registered kernel:

  FULL_SOFTMAX : standard scaled_dot_product_attention (Q heads × full KV)
  WINDOW_32    : native windowed O(n·w) — slices K,V to last 32 positions
  BOS_ROUTE    : attends only to BOS (pos 0) + last 8 tokens (9-token KV budget)

NOTE on implementation fidelity:
  WINDOW_32 computes scaled_dot_product_attention on K[:,-32:,:] and V[:,-32:,:].
  This materializes only a (head_dim × 32) KV matrix, not the full (head_dim × n)
  one, so it is genuinely O(n·w) in memory and compute.  Logit masking is NOT used.
"""

import json, os, torch, copy
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL      = "Qwen/Qwen2.5-0.5B"
SAFE_MODEL = MODEL.split("/")[-1]
ROUTING_FILE = f"outputs/routing/{SAFE_MODEL}_stability.json"

WINDOW_SIZE = 32
BOS_BUDGET  = 8   # BOS + last 8 tokens

print(f"Loading routing map from {ROUTING_FILE}...")
with open(ROUTING_FILE) as f:
    stab = json.load(f)
routing_map = stab["routing_map"]   # {"L_H": {"routing": "WINDOW_32", ...}}

def parse_key(k):
    l, h = k.split("_")
    return int(l), int(h)

# Organise per-layer routing decisions for fast lookup
layer_routing = {}  # {layer_idx: {head_idx: "WINDOW_32" | "BOS_ROUTE" | "FULL_SOFTMAX"}}
for k, v in routing_map.items():
    l, h = parse_key(k)
    layer_routing.setdefault(l, {})[h] = v["routing"]

def make_routed_forward(original_forward, layer_idx, config, layer_routing_map):
    """
    Returns a new forward() that replaces the attention computation
    for this specific layer, dispatching per head.
    """
    num_heads    = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim     = config.hidden_size // num_heads
    kv_group     = num_heads // num_kv_heads   # heads per KV group

    def routed_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        attn_module = original_forward.__self__

        # ── Q, K, V projections (unchanged) ──────────────────────────────
        query_states = attn_module.q_proj(hidden_states)
        key_states   = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        # Reshape to (bsz, n_heads, seq, head_dim)
        query_states = query_states.view(bsz, q_len, num_heads,    head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE if present
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ── Per-head dispatch ─────────────────────────────────────────────
        head_outputs = []
        for h in range(num_heads):
            kv_h = h // kv_group       # which KV head this Q head uses
            q_h  = query_states[:, h:h+1, :, :]      # (bsz, 1, q_len, head_dim)
            k_h  = key_states[:, kv_h:kv_h+1, :, :]  # (bsz, 1, seq, head_dim)
            v_h  = value_states[:, kv_h:kv_h+1, :, :]

            route = layer_routing_map.get(h, "FULL_SOFTMAX")

            if route == "WINDOW_32":
                # Native windowed: O(n·w) sliding window
                # During prefill (q_len > 1), we chunk over queries to avoid materializing full KV mask
                if q_len == 1: # decode phase
                    k_w = k_h[:, :, max(0, k_h.shape[2] - WINDOW_SIZE):, :]
                    v_w = v_h[:, :, max(0, v_h.shape[2] - WINDOW_SIZE):, :]
                    out = F.scaled_dot_product_attention(q_h, k_w, v_w, is_causal=False)
                else: # prefill phase
                    outs = []
                    # Loop over query positions to compute sliding window exactly O(n*w)
                    for i in range(q_len):
                        q_i = q_h[:, :, i:i+1, :]
                        start = max(0, i + 1 - WINDOW_SIZE)
                        end = i + 1
                        k_i = k_h[:, :, start:end, :]
                        v_i = v_h[:, :, start:end, :]
                        outs.append(F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False))
                    out = torch.cat(outs, dim=2)

            elif route == "BOS_ROUTE":
                # Keep only BOS (pos 0) + last BOS_BUDGET tokens (relative to query)
                if q_len == 1:
                    seq = k_h.shape[2]
                    bos_k, bos_v = k_h[:, :, :1, :], v_h[:, :, :1, :]
                    if seq > 1:
                        tail_k = k_h[:, :, max(1, seq - BOS_BUDGET):, :]
                        tail_v = v_h[:, :, max(1, seq - BOS_BUDGET):, :]
                        k_w = torch.cat([bos_k, tail_k], dim=2)
                        v_w = torch.cat([bos_v, tail_v], dim=2)
                    else:
                        k_w, v_w = bos_k, bos_v
                    out = F.scaled_dot_product_attention(q_h, k_w, v_w, is_causal=False)
                else:
                    outs = []
                    bos_k, bos_v = k_h[:, :, :1, :], v_h[:, :, :1, :]
                    for i in range(q_len):
                        q_i = q_h[:, :, i:i+1, :]
                        end = i + 1
                        if end > 1:
                            start_tail = max(1, end - BOS_BUDGET)
                            tail_k = k_h[:, :, start_tail:end, :]
                            tail_v = v_h[:, :, start_tail:end, :]
                            k_i = torch.cat([bos_k, tail_k], dim=2)
                            v_i = torch.cat([bos_v, tail_v], dim=2)
                        else:
                            k_i, v_i = bos_k, bos_v
                        outs.append(F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False))
                    out = torch.cat(outs, dim=2)

            else:  # FULL_SOFTMAX — full causal
                # Build a causal mask for this head
                causal = torch.ones(q_len, k_h.shape[2], dtype=torch.bool, device=q_h.device).tril()
                out = F.scaled_dot_product_attention(
                    q_h, k_h, v_h,
                    attn_mask=causal,
                )

            head_outputs.append(out)

        # Concatenate heads and project
        attn_output = torch.cat(head_outputs, dim=1)               # (bsz, H, q_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()     # (bsz, q_len, H, head_dim)
        attn_output = attn_output.reshape(bsz, q_len, -1)          # (bsz, q_len, hidden)
        attn_output = attn_module.o_proj(attn_output)

        return attn_output, None  # (hidden_states, attn_weights) — matches decoder's 'hidden_states, _ = self.self_attn(...)'

    routed_forward.__self__ = original_forward.__self__
    return routed_forward

def patch_model(model):
    """Patch all attention layers with routing-aware forward."""
    num_layers = model.config.num_hidden_layers
    for l in range(num_layers):
        lr = layer_routing.get(l, {})
        if not lr:
            continue  # no routing decisions for this layer, leave untouched
        attn = model.model.layers[l].self_attn
        attn.forward = make_routed_forward(attn.forward, l, model.config, lr)
    print(f"Patched {num_layers} attention layers with routing engine.")
    return model

if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
    model.eval()

    text = "The quick brown fox jumps over the lazy dog."
    ids  = tok(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out_base = model(**ids)
        logits_base = out_base.logits[0, -1].clone()

    patch_model(model)

    with torch.no_grad():
        out_routed = model(**ids)
        logits_routed = out_routed.logits[0, -1]

    delta = (logits_base - logits_routed).abs().max().item()
    print(f"Max logit delta (sanity check): {delta:.4f}")
    top_base   = logits_base.argmax().item()
    top_routed = logits_routed.argmax().item()
    print(f"Top token baseline: {tok.decode([top_base])!r}")
    print(f"Top token routed:   {tok.decode([top_routed])!r}")
