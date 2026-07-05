"""
step4_softmax_saturation.py  (Behavioral Law 11 — Softmax Saturation)
----------------------------------------------------------------------
For each head, measure the max attention weight in the attention row.
A head operating as a "hard gate" (Retrieval) should have near-1.0 max weight.
A head operating in a distributed regime (Local/Sink) should have lower max weight.

Metrics per head:
  - mean_max_attn : mean over prompts of max(attention row)
  - mean_entropy  : mean attention entropy (cross-check with Phase 1 data)

Output: outputs/phase2_atlas/softmax_saturation.json
"""

import json, os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
SAFE_MODEL = MODEL.split("/")[-1]
DATASET = "outputs/phase2_atlas/dataset.json"
OUT     = f"outputs/phase2_atlas/{SAFE_MODEL}_softmax_saturation.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = AutoTokenizer.from_pretrained(MODEL)
model  = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

with open(DATASET) as f:
    data = json.load(f)
texts = [s["text"] for s in data["wikitext"]]

L = model.config.num_hidden_layers
H = model.config.num_attention_heads
acc_max_attn = {(l, h): [] for l in range(L) for h in range(H)}
acc_entropy  = {(l, h): [] for l in range(L) for h in range(H)}

print(f"Running {len(texts)} prompts...")
for i, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    if ids["input_ids"].shape[1] < 4:
        continue
    with torch.no_grad():
        out = model(**ids, output_attentions=True)

    for l, attn in enumerate(out.attentions):
        a = attn[0].float().cpu().numpy()  # (H, T, T)
        for h in range(H):
            row = a[h, -1, :]  # last token row
            # max attention weight
            acc_max_attn[(l, h)].append(float(row.max()))
            # entropy
            p = row + 1e-12
            p = p / p.sum()
            ent = float(-np.sum(p * np.log(p)))
            acc_entropy[(l, h)].append(ent)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(texts)} done")

results = {}
for l in range(L):
    for h in range(H):
        key = f"{l}_{h}"
        results[key] = {
            "mean_max_attn": round(float(np.mean(acc_max_attn[(l, h)])), 4),
            "mean_entropy":  round(float(np.mean(acc_entropy[(l, h)])),  4),
        }

out_data = {"model": MODEL, "heads": results}
with open(OUT, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved to {OUT}")
