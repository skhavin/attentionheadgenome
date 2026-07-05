"""
step6_compile_atlas.py  (Final — Assemble the Head Atlas)
----------------------------------------------------------
Reads all outputs from steps 1–5 and compiles one unified JSON
"atlas" card per head, matching the schema defined in plan.md.

Output: outputs/phase2_atlas/{MODEL}_head_atlas.json
"""

import json, os, sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
SAFE_MODEL = MODEL.split("/")[-1]

PHASE1_FILE = {
    "gpt2-medium": "outputs/phase1/robust_entropy_gpt2.json",
    "Qwen2.5-0.5B": "outputs/phase1/robust_entropy_qwen0.5b.json",
    "Qwen2.5-1.5B": "outputs/phase1/robust_entropy_qwen1.5b.json",
    "Llama-3.2-1B": "outputs/phase1/llama1b_retrieval_entropy.json",
}.get(SAFE_MODEL, None)

DIST_FILE   = f"outputs/phase2_atlas/{SAFE_MODEL}_distance_profile.json"
OV_FILE     = f"outputs/phase2_atlas/{SAFE_MODEL}_ov_output_norm.json"
GRAM_FILE   = f"outputs/phase2_atlas/{SAFE_MODEL}_grammar_map.json"
SAT_FILE    = f"outputs/phase2_atlas/{SAFE_MODEL}_softmax_saturation.json"
SINK_FILE   = f"outputs/phase2_atlas/{SAFE_MODEL}_sink_falsification.json"
OUT         = f"outputs/phase2_atlas/{SAFE_MODEL}_head_atlas.json"

def load(path):
    if not path or not os.path.exists(path):
        print(f"  WARNING: missing {path}, skipping.")
        return None
    with open(path) as f:
        return json.load(f)

p1   = load(PHASE1_FILE)
dist = load(DIST_FILE)
ov   = load(OV_FILE)
gram = load(GRAM_FILE)
sat  = load(SAT_FILE)
sink = load(SINK_FILE)

RETRIEVAL_THRESH =  0.30
INDUCTION_THRESH = -0.50

def classify(delta):
    if delta >= RETRIEVAL_THRESH:
        return "Retrieval"
    if delta <= INDUCTION_THRESH:
        return "Induction"
    return "Local"

atlas = {}

# Use dist file as the source of truth for all heads, since it runs on all models
if not dist:
    print(f"CRITICAL: Distance profile missing for {SAFE_MODEL}. Cannot compile atlas.")
    sys.exit(1)

for key in dist["heads"].keys():
    l_str, h_str = key.split("_")
    l, h = int(l_str), int(h_str)

    delta = 0.0
    match_ent = None
    nonmatch_ent = None
    label = "Local"

    if p1 and key in p1["heads"]:
        v = p1["heads"][key]
        delta = v.get("delta", 0.0)
        match_ent = v.get("match_entropy")
        nonmatch_ent = v.get("nonmatch_entropy")
        label = classify(delta)
    elif sink and key in sink["heads"]:
        # Fallback: if we don't have Phase 1 delta, use the step5 baseline entropy
        pass

    # Upgrade to Sink if BOS mass is very high
    bos_mass = dist["heads"][key].get("bos_mass", 0.0)
    if bos_mass > 0.50:
        label = "Sink"

    card = {
        "model":       SAFE_MODEL,
        "layer":       l,
        "head":        h,
        "class_label": label,
        "entropy_profile": {
            "match_entropy":    match_ent,
            "nonmatch_entropy": nonmatch_ent,
            "delta_collapse":   delta,
        },
        "attention_geometry": dist["heads"][key],
    }

    if ov and key in ov["heads"]:
        d = ov["heads"][key]
        card["vq_ratio"]        = d.get("vq_ratio")
        card["mean_output_norm"] = d.get("mean_output_norm")

    if gram and key in gram["heads"]:
        card["grammar_profile"] = gram["heads"][key]

    if sat and key in sat["heads"]:
        card["softmax_saturation"] = sat["heads"][key]

    if sink and key in sink["heads"]:
        card["sink_falsification"] = sink["heads"][key]

    atlas[key] = card

out_data = {
    "model":        SAFE_MODEL,
    "total_heads":  len(atlas),
    "thresholds": {
        "retrieval": RETRIEVAL_THRESH,
        "induction": INDUCTION_THRESH,
        "sink_bos_mass": 0.50,
    },
    "heads": atlas,
}

with open(OUT, "w") as f:
    json.dump(out_data, f, indent=2)

# Print summary
label_counts = {}
for card in atlas.values():
    lbl = card["class_label"]
    label_counts[lbl] = label_counts.get(lbl, 0) + 1

print(f"Atlas compiled in {OUT}")
print(f"  Total heads: {len(atlas)}")
for lbl, cnt in sorted(label_counts.items()):
    print(f"  {lbl}: {cnt}")
