"""
step2_ov_output_norm.py  (Pillar 2 — OV Circuit / Write Effect)
---------------------------------------------------------------
For every head, compute the output norm ||O_h|| — the magnitude
of what it writes into the residual stream — across the 100 WikiText prompts.

Also computes the V/Q structural ratio from the model's weight matrices:
  vq_ratio = ||W_V||_F / ||W_Q||_F

Output: outputs/phase2_atlas/{MODEL}_ov_output_norm.json
"""

import json, os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
SAFE_MODEL = MODEL.split("/")[-1]
DATASET = "outputs/phase2_atlas/dataset.json"
OUT     = f"outputs/phase2_atlas/{SAFE_MODEL}_ov_output_norm.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = AutoTokenizer.from_pretrained(MODEL)
model  = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

with open(DATASET) as f:
    data = json.load(f)
texts = [s["text"] for s in data["wikitext"]]

L = model.config.num_hidden_layers
H = model.config.num_attention_heads
head_dim = model.config.hidden_size // H

num_kv_heads = getattr(model.config, "num_key_value_heads", H)
kv_groups = H // num_kv_heads

# ── Structural V/Q ratios from weight matrices ────────────────────────────────
vq_ratios = {}
for l in range(L):
    if "gpt2" in MODEL.lower():
        blk = model.transformer.h[l].attn
        w = blk.c_attn.weight.T  # (3*n_embd, n_embd) after transpose
        n_embd = model.config.n_embd
        w_q = w[:n_embd, :]
        w_v = w[2*n_embd:, :]
        for h in range(H):
            s, e = h * head_dim, (h + 1) * head_dim
            norm_q = float(w_q[s:e, :].norm(p="fro"))
            norm_v = float(w_v[s:e, :].norm(p="fro"))
            vq_ratios[(l, h)] = round(norm_v / norm_q, 4) if norm_q > 0 else 0.0
    else:
        # Llama or Qwen
        blk = model.model.layers[l].self_attn
        w_q = blk.q_proj.weight  # (H * head_dim, hidden_size)
        w_v = blk.v_proj.weight  # (num_kv_heads * head_dim, hidden_size)
        for h in range(H):
            kv_idx = h // kv_groups
            s_q, e_q = h * head_dim, (h + 1) * head_dim
            s_v, e_v = kv_idx * head_dim, (kv_idx + 1) * head_dim
            norm_q = float(w_q[s_q:e_q, :].norm(p="fro"))
            norm_v = float(w_v[s_v:e_v, :].norm(p="fro"))
            vq_ratios[(l, h)] = round(norm_v / norm_q, 4) if norm_q > 0 else 0.0

# ── Runtime output norms via hooks ────────────────────────────────────────────
head_output_norms = {(l, h): [] for l in range(L) for h in range(H)}

def make_hook():
    def hook(module, input, output):
        module._last_output = output
    return hook

hooks = []
for l in range(L):
    if "gpt2" in MODEL.lower():
        h_obj = model.transformer.h[l].attn
    else:
        h_obj = model.model.layers[l].self_attn
    hooks.append(h_obj.register_forward_hook(make_hook()))

print(f"Running {len(texts)} prompts on {MODEL}...")
for i, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    if ids["input_ids"].shape[1] < 4:
        continue
    with torch.no_grad():
        model(**ids, output_attentions=True)

    for l in range(L):
        if "gpt2" in MODEL.lower():
            attn_module = model.transformer.h[l].attn
        else:
            attn_module = model.model.layers[l].self_attn
            
        out_tensor = attn_module._last_output[0]  # (1, T, n_embd)
        proxy_norm = float(out_tensor[0, -1, :].norm().cpu()) / (H ** 0.5)
        for h in range(H):
            head_output_norms[(l, h)].append(proxy_norm)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(texts)} done")

for hook in hooks:
    hook.remove()

# ── Assemble output ────────────────────────────────────────────────────────────
results = {}
for l in range(L):
    for h in range(H):
        key = f"{l}_{h}"
        norms = head_output_norms[(l, h)]
        results[key] = {
            "vq_ratio":        vq_ratios[(l, h)],
            "mean_output_norm": round(float(np.mean(norms)), 4) if norms else 0.0,
        }

out_data = {"model": SAFE_MODEL, "heads": results}
with open(OUT, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved to {OUT}")
