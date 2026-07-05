"""
step5_sink_falsification.py  (Pillar 4 — Sink Head Falsification)
-----------------------------------------------------------------
Tests the purpose of Sink heads by comparing:
  - Baseline: normal prompt
  - Condition A: BOS token removed (prepend a neutral padding token instead)
  - Condition B: BOS replaced with a random mid-vocab token (id=500)

For each condition, measure the entropy of every head.
If Sink heads exist to absorb unused attention via BOS, their entropy
should increase dramatically when BOS is removed.

Uses the 100 WikiText prompts.

Output: outputs/phase2_atlas/sink_falsification.json
Schema: {
  "model": str,
  "heads": {
    "L_H": {
      "entropy_baseline": float,
      "entropy_no_bos": float,
      "entropy_replaced_bos": float,
      "delta_no_bos": float,       # entropy_no_bos - entropy_baseline
      "delta_replaced_bos": float
    }
  }
}
"""

import json, os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
SAFE_MODEL = MODEL.split("/")[-1]
DATASET = "outputs/phase2_atlas/dataset.json"
OUT     = f"outputs/phase2_atlas/{SAFE_MODEL}_sink_falsification.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = AutoTokenizer.from_pretrained(MODEL)
model  = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

with open(DATASET) as f:
    data = json.load(f)
texts = [s["text"] for s in data["wikitext"]]

L = model.config.num_hidden_layers
H = model.config.num_attention_heads
REPLACEMENT_TOKEN_ID = 500  # a mid-vocab token unlikely to be BOS

def head_entropies(input_ids):
    """Return list of length L, each element array of shape (H,) with entropy per head."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)
    entropies = []
    for attn in out.attentions:
        a = attn[0].float().cpu().numpy()  # (H, T, T)
        row = a[:, -1, :]  # (H, T) - last token
        p = row + 1e-12
        p = p / p.sum(axis=-1, keepdims=True)
        ent = -np.sum(p * np.log(p), axis=-1)  # (H,)
        entropies.append(ent)
    return entropies  # list of L arrays of shape (H,)

acc = {(l, h): {"base": [], "no_bos": [], "rep_bos": []}
       for l in range(L) for h in range(H)}

print(f"Running {len(texts)} prompts (3 conditions each)...")
for i, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device)
    T = ids.shape[1]
    if T < 5:
        continue

    # Baseline
    ent_base = head_entropies(ids)

    # Condition A: remove BOS (drop first token)
    ids_no_bos = ids[:, 1:]
    if ids_no_bos.shape[1] < 3:
        continue
    ent_no_bos = head_entropies(ids_no_bos)

    # Condition B: replace BOS with random token
    ids_rep = ids.clone()
    ids_rep[0, 0] = REPLACEMENT_TOKEN_ID
    ent_rep = head_entropies(ids_rep)

    for l in range(L):
        for h in range(H):
            acc[(l, h)]["base"].append(float(ent_base[l][h]))
            acc[(l, h)]["no_bos"].append(float(ent_no_bos[l][h]))
            acc[(l, h)]["rep_bos"].append(float(ent_rep[l][h]))

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(texts)} done")

results = {}
for l in range(L):
    for h in range(H):
        key = f"{l}_{h}"
        base = float(np.mean(acc[(l, h)]["base"]))
        no_b = float(np.mean(acc[(l, h)]["no_bos"]))
        rep  = float(np.mean(acc[(l, h)]["rep_bos"]))
        results[key] = {
            "entropy_baseline":     round(base, 4),
            "entropy_no_bos":       round(no_b, 4),
            "entropy_replaced_bos": round(rep,  4),
            "delta_no_bos":         round(no_b - base, 4),
            "delta_replaced_bos":   round(rep  - base, 4),
        }

out_data = {"model": MODEL, "heads": results}
with open(OUT, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved to {OUT}")
