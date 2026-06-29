"""
canonical_classification.py
============================
Single-source-of-truth classifier. Uses:
  - gpt2_mechanistic_labels.json as the AUTHORITATIVE canonical source for GPT-2
    (thresholds: retrieval_delta>=0.3, induction_delta<=-0.5, sink_entropy<=0.1)
  - The same thresholds applied to robust_entropy_{model}.json for Qwen/Llama.

For Llama, there are NO sink heads above the entropy threshold in the data —
this is a real finding (Llama uses RoPE so there's no APE-style BOS sink).

Outputs:
  outputs/canonical_labels.json  — the single canonical source for ALL figures.
"""
import json
import os
from collections import Counter

RETRIEVAL_DELTA_MIN  =  0.3    # entropy drop > +0.3 on retrieval task
INDUCTION_DELTA_MAX  = -0.5    # entropy drop < -0.5 on induction task
SINK_ENTROPY_MAX     =  0.1    # match_entropy < 0.1 → always-collapsed BOS sink

# ── GPT-2: use authoritative pre-labelled file ──────────────────────────────
gpt2_ref = json.load(open("outputs/phase1/gpt2_mechanistic_labels.json"))
gpt2_heads_raw = json.load(open("outputs/phase1/robust_entropy_gpt2.json"))["heads"]

gpt2_heads = {}
for head_id, label in gpt2_ref["heads"].items():
    layer = int(head_id.split("_")[0])
    info  = gpt2_heads_raw.get(head_id, {})
    gpt2_heads[head_id] = {
        "layer":          layer,
        "head_idx":       int(head_id.split("_")[1]),
        "label":          label,
        "relative_depth": layer / 24,
        "delta":          info.get("delta", 0.0),
        "match_entropy":  info.get("match_entropy", 0.0),
    }

print(f"GPT-2: total={len(gpt2_heads)}, {dict(Counter(v['label'] for v in gpt2_heads.values()))}")
assert Counter(v["label"] for v in gpt2_heads.values())["local"] == gpt2_ref["counts"]["local"], \
    "GPT-2 local count mismatch!"

# ── Qwen / Llama: apply thresholds to robust_entropy files ──────────────────
other_files = {
    "Qwen-0.5B":   ("outputs/phase1/robust_entropy_qwen0.5b.json",  24),
    "Qwen-1.5B":   ("outputs/phase1/robust_entropy_qwen1.5b.json",  28),
    "Llama-3.2-1B":("outputs/phase1/llama1b_retrieval_entropy.json", 16),
}

def classify_head(delta, match_entropy):
    if match_entropy < SINK_ENTROPY_MAX:
        return "sink"
    if delta >= RETRIEVAL_DELTA_MIN:
        return "retrieval"
    if delta <= INDUCTION_DELTA_MAX:
        return "induction"
    return "local"

canonical = {
    "GPT-2": {"n_layers": 24, "heads": gpt2_heads, "counts": dict(Counter(v["label"] for v in gpt2_heads.values()))}
}

for model, (path, n_layers) in other_files.items():
    raw = json.load(open(path))["heads"]
    heads = {}
    for head_id, info in raw.items():
        layer = int(head_id.split("_")[0])
        label = classify_head(info.get("delta", 0.0), info.get("match_entropy", 999.0))
        heads[head_id] = {
            "layer":          layer,
            "head_idx":       int(head_id.split("_")[1]),
            "label":          label,
            "relative_depth": layer / n_layers,
            "delta":          info.get("delta", 0.0),
            "match_entropy":  info.get("match_entropy", 0.0),
        }
    counts = dict(Counter(v["label"] for v in heads.values()))
    print(f"{model}: total={len(heads)}, {counts}")
    canonical[model] = {"n_layers": n_layers, "heads": heads, "counts": counts}

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
output = {
    "thresholds": {
        "retrieval_delta_min": RETRIEVAL_DELTA_MIN,
        "induction_delta_max": INDUCTION_DELTA_MAX,
        "sink_entropy_max":    SINK_ENTROPY_MAX,
        "note": "GPT-2 labels sourced from gpt2_mechanistic_labels.json (authoritative). Others derived from robust_entropy JSON files using identical thresholds."
    },
    "models": canonical
}
with open("outputs/canonical_labels.json", "w") as f:
    json.dump(output, f, indent=2)

# ── Summary table ────────────────────────────────────────────────────────────
print("\n=== Canonical Classification Summary ===")
total = Counter()
for model, data in canonical.items():
    c = data["counts"]
    total.update(c)
    print(f"  {model:15}: Sink={c.get('sink',0):3d}  Local={c.get('local',0):4d}  "
          f"Retrieval={c.get('retrieval',0):3d}  Induction={c.get('induction',0):3d}  "
          f"Total={sum(c.values())}")
print(f"  {'TOTAL':15}: Sink={total.get('sink',0):3d}  Local={total.get('local',0):4d}  "
      f"Retrieval={total.get('retrieval',0):3d}  Induction={total.get('induction',0):3d}  "
      f"Total={sum(total.values())}")
print("\nSaved: outputs/canonical_labels.json")
