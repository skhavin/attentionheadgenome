"""
step1_distance_profile.py  (Pillar 1 — Attention Target Geometry)
------------------------------------------------------------------
For every head, measure the distribution of attended distances (t - j)
across the 100 WikiText prompts.

For each head we record:
  - mean_distance   : average relative position of attended token
  - bos_mass        : fraction of attention that lands on token 0
  - local_mass      : fraction within distance 1..4
  - long_range_mass : fraction > distance 32

Output: outputs/phase2_atlas/distance_profile.json
Schema: { "model": str, "heads": { "L_H": {mean_distance, bos_mass, local_mass, long_range_mass} } }
"""

import json, os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
SAFE_MODEL = MODEL.split("/")[-1]
DATASET = "outputs/phase2_atlas/dataset.json"
OUT     = f"outputs/phase2_atlas/{SAFE_MODEL}_distance_profile.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = AutoTokenizer.from_pretrained(MODEL)
model  = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

with open(DATASET) as f:
    data = json.load(f)

texts = [s["text"] for s in data["wikitext"]]

L = model.config.num_hidden_layers
H = model.config.num_attention_heads
# accumulators: per head, list of (mean_dist, bos_mass, local_mass, long_range_mass) across prompts
acc = {(l, h): {"dist": [], "bos": [], "local": [], "long": []} for l in range(L) for h in range(H)}

print(f"Running {len(texts)} prompts through {MODEL}...")
for i, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    T   = ids["input_ids"].shape[1]
    if T < 4:
        continue

    with torch.no_grad():
        out = model(**ids, output_attentions=True)

    # out.attentions: tuple of (1, H, T, T) per layer
    for l, attn in enumerate(out.attentions):
        a = attn[0].float().cpu().numpy()  # (H, T, T)
        for h in range(H):
            # use the last token's attention row as representative
            row = a[h, -1, :]              # (T,)
            t   = len(row) - 1             # position of last token
            positions = np.arange(T)
            dists     = t - positions      # distance from each source to last token

            bos_m   = float(row[0])
            local_m = float(row[max(0, t-4):t].sum())
            long_m  = float(row[: max(0, t-32)].sum()) if t > 32 else 0.0
            mean_d  = float((row * dists).sum())

            acc[(l, h)]["dist"].append(mean_d)
            acc[(l, h)]["bos"].append(bos_m)
            acc[(l, h)]["local"].append(local_m)
            acc[(l, h)]["long"].append(long_m)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(texts)} done")

# Aggregate
results = {}
for (l, h), v in acc.items():
    results[f"{l}_{h}"] = {
        "mean_distance":   round(float(np.mean(v["dist"])), 4),
        "bos_mass":        round(float(np.mean(v["bos"])),  4),
        "local_mass":      round(float(np.mean(v["local"])),4),
        "long_range_mass": round(float(np.mean(v["long"])), 4),
    }

out_data = {"model": MODEL, "heads": results}
with open(OUT, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved to {OUT}")
