# phase1/step9_retrieval_entropy_qwen.py
#
# PURPOSE:
#   Run the synthetic entropy-collapse experiment on Qwen2.5-0.5B (GQA).
#   Measures per-head entropy-collapse delta on matching vs. non-matching prompts.
#
# GQA-SPECIFIC ADDITION:
#   Qwen2.5-0.5B: 14 Q heads, 2 KV heads, group_size=7.
#   For every retrieval candidate found, record its KV group.
#   Hypothesis: if retrieval heads exist in GQA, they cluster within KV groups
#   because the K projection is shared — one retrieval-capable K serves 7 Q heads.
#
# OUTPUTS:
#   outputs/phase1/qwen0.5b_retrieval_entropy.json

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")

MODEL_ID   = "Qwen/Qwen2.5-0.5B"
MODEL_SLUG = "qwen0.5b"

# Mechanistic thresholds (same as GPT-2 for cross-model consistency)
THRESHOLD_RETRIEVAL = 0.30
THRESHOLD_INDUCTION = -0.50
THRESHOLD_SINK_ENT  = 0.10

# Same 20 prompt pairs as GPT-2 step7 (identical stimuli = clean comparison)
PROMPT_PAIRS = [
    ("The capital of France is Paris.",
     "The weather today is sunny and warm.",
     " The capital of France is"),
    ("The speed of light is 299792458 meters per second.",
     "The dog ran quickly across the field.",
     " The speed of light is"),
    ("Shakespeare was born in Stratford-upon-Avon.",
     "The stock market closed higher yesterday.",
     " Shakespeare was born in"),
    ("Water boils at 100 degrees Celsius.",
     "The train arrived at the station late.",
     " Water boils at"),
    ("The Eiffel Tower is located in Paris.",
     "She enjoyed reading books in the evening.",
     " The Eiffel Tower is located in"),
    ("Mount Everest is the tallest mountain on Earth.",
     "He enjoyed cooking pasta for dinner.",
     " Mount Everest is the tallest"),
    ("The chemical formula for water is H2O.",
     "The children played outside all afternoon.",
     " The chemical formula for water is"),
    ("Leonardo da Vinci painted the Mona Lisa.",
     "The conference was held in a large hall.",
     " Leonardo da Vinci painted the"),
    ("The Great Wall of China was built over many centuries.",
     "The new restaurant opened last week downtown.",
     " The Great Wall of China was"),
    ("Albert Einstein developed the theory of relativity.",
     "The supermarket was crowded on Saturday morning.",
     " Albert Einstein developed the theory of"),
    ("Oxygen has the atomic number 8.",
     "The library closed early due to renovations.",
     " Oxygen has the atomic number"),
    ("The Amazon River flows through Brazil.",
     "She spent the afternoon painting in her studio.",
     " The Amazon River flows through"),
    ("The human body has 206 bones.",
     "The cat curled up on the warm windowsill.",
     " The human body has"),
    ("Isaac Newton discovered gravity.",
     "They drove through the mountains on their vacation.",
     " Isaac Newton discovered"),
    ("The Pacific Ocean is the largest ocean on Earth.",
     "He forgot to bring his umbrella to work.",
     " The Pacific Ocean is the"),
    ("DNA stands for deoxyribonucleic acid.",
     "The children built a sandcastle on the beach.",
     " DNA stands for"),
    ("Rome is the capital of Italy.",
     "She planted roses in her garden this spring.",
     " Rome is the capital of"),
    ("The Berlin Wall fell in 1989.",
     "He enjoyed hiking in the national park.",
     " The Berlin Wall fell in"),
    ("The Pythagorean theorem states that a squared plus b squared equals c squared.",
     "She read a mystery novel before going to sleep.",
     " The Pythagorean theorem states"),
    ("Photosynthesis converts sunlight into chemical energy.",
     "They watched the fireworks from the hilltop.",
     " Photosynthesis converts sunlight into"),
]


def attention_entropy(attn_weights):
    """
    attn_weights: (num_q_heads, seq_len, seq_len)
    Returns: (num_q_heads,) entropy at the last query position.
    Cast to float32 to avoid float16 underflow: 1e-12 rounds to 0 in fp16.
    """
    last_pos = attn_weights[:, -1, :].float()   # (num_q_heads, seq_len)
    p = last_pos + 1e-12
    p = p / p.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(p * torch.log(p), dim=-1)
    return entropy.cpu().numpy()


def run_prompt(model, tokenizer, text, device):
    """
    Returns list[layer] of (num_q_heads,) entropy arrays.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    layer_entropies = []
    for attn in out.attentions:
        if attn is None:
            raise RuntimeError("Attention is None — model must use attn_implementation='eager'.")
        layer_entropies.append(attention_entropy(attn[0]))
    return layer_entropies


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    print("Loading " + MODEL_ID + " (attn_implementation=eager)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if device == "cuda":
        model = model.half().cuda()
    model.eval()

    num_layers  = model.config.num_hidden_layers    # 24
    num_q_heads = model.config.num_attention_heads  # 14
    num_kv_heads= model.config.num_key_value_heads  # 2
    gqa_group   = num_q_heads // num_kv_heads       # 7

    print("Architecture: " + str(num_layers) + " layers, " +
          str(num_q_heads) + " Q heads, " +
          str(num_kv_heads) + " KV heads, GQA group=" + str(gqa_group))

    # Accumulate entropy per (layer, head)
    match_ent    = [[[] for _ in range(num_q_heads)] for _ in range(num_layers)]
    nonmatch_ent = [[[] for _ in range(num_q_heads)] for _ in range(num_layers)]

    print("Running " + str(len(PROMPT_PAIRS)) + " prompt pairs...")
    for i, (ctx_m, ctx_nm, query) in enumerate(PROMPT_PAIRS):
        me  = run_prompt(model, tokenizer, ctx_m  + query, device)
        nme = run_prompt(model, tokenizer, ctx_nm + query, device)

        for l in range(num_layers):
            for h in range(num_q_heads):
                match_ent[l][h].append(float(me[l][h]))
                nonmatch_ent[l][h].append(float(nme[l][h]))

        if (i + 1) % 5 == 0:
            print("  Processed " + str(i + 1) + "/" + str(len(PROMPT_PAIRS)) + " pairs")

    # Compute deltas
    print("\nComputing deltas...")
    results    = {}
    all_deltas = []
    nan_heads  = []

    for l in range(num_layers):
        for h in range(num_q_heads):
            me_val  = float(np.mean(match_ent[l][h]))
            nme_val = float(np.mean(nonmatch_ent[l][h]))
            delta   = nme_val - me_val
            key     = str(l) + "_" + str(h)
            kv_grp  = h // gqa_group  # which KV head this Q head belongs to

            if np.isnan(delta) or np.isnan(me_val):
                nan_heads.append((l, h, kv_grp))
                results[key] = {"match_entropy": None, "nonmatch_entropy": None,
                                "delta": None, "kv_group": kv_grp, "nan": True}
                delta = 0.0
            else:
                results[key] = {
                    "match_entropy":    round(me_val, 5),
                    "nonmatch_entropy": round(nme_val, 5),
                    "delta":            round(delta, 5),
                    "kv_group":         kv_grp,
                }
            all_deltas.append((delta, l, h, kv_grp))

    all_deltas.sort(reverse=True)

    # Apply mechanistic labels
    labels = {}
    counts = {"sink": 0, "retrieval": 0, "induction": 0, "local": 0}
    for key, v in results.items():
        if v.get("nan") or v["match_entropy"] is None:
            role = "sink"
        elif v["match_entropy"] < THRESHOLD_SINK_ENT and v["nonmatch_entropy"] < THRESHOLD_SINK_ENT:
            role = "sink"
        elif v["delta"] > THRESHOLD_RETRIEVAL:
            role = "retrieval"
        elif v["delta"] < THRESHOLD_INDUCTION:
            role = "induction"
        else:
            role = "local"
        labels[key] = role
        counts[role] += 1

    # ── Print results ────────────────────────────────────────────────────────
    total = sum(counts.values())
    print("\n=== Qwen2.5-0.5B Mechanistic Labels ===")
    for role in ["sink", "retrieval", "induction", "local"]:
        n = counts[role]
        pct = (n / total) * 100
        print("  " + role.ljust(12) + str(n).rjust(4) + "  (" + str(round(pct, 1)) + "%)")

    print("\n=== Top 15 Entropy-Collapse Heads (retrieval candidates) ===")
    print("  rank  layer  head  kv_grp  delta     match_ent  nonmatch_ent")
    for rank, (delta, l, h, kv_grp) in enumerate(all_deltas[:15]):
        key = str(l) + "_" + str(h)
        me_v  = results[key].get("match_entropy", "NaN")
        nme_v = results[key].get("nonmatch_entropy", "NaN")
        print("  " + str(rank+1).ljust(6) +
              str(l).ljust(7) + str(h).ljust(6) +
              str(kv_grp).ljust(8) +
              str(round(delta, 4)).ljust(10) +
              str(round(me_v, 4) if me_v else "NaN").ljust(11) +
              str(round(nme_v, 4) if nme_v else "NaN"))

    print("\n=== Bottom 10 Heads (no retrieval) ===")
    print("  rank  layer  head  kv_grp  delta")
    for rank, (delta, l, h, kv_grp) in enumerate(all_deltas[-10:]):
        actual_rank = total - 10 + rank + 1
        print("  " + str(actual_rank).ljust(6) +
              str(l).ljust(7) + str(h).ljust(6) +
              str(kv_grp).ljust(8) +
              str(round(delta, 4)))

    # Distribution
    valid_deltas = np.array([d for d, *_ in all_deltas if d != 0.0 or True])
    # Filter out sentinel 0.0 from NaN heads
    nan_set = set(str(l) + "_" + str(h) for l, h, _ in nan_heads)
    clean_deltas = np.array([d for d, l, h, _ in all_deltas
                              if str(l) + "_" + str(h) not in nan_set])
    print("\n=== Delta Distribution (excluding NaN/sink heads) ===")
    print("  mean:  " + str(round(float(clean_deltas.mean()), 5)))
    print("  std:   " + str(round(float(clean_deltas.std()), 5)))
    print("  p75:   " + str(round(float(np.percentile(clean_deltas, 75)), 5)))
    print("  p90:   " + str(round(float(np.percentile(clean_deltas, 90)), 5)))
    print("  p95:   " + str(round(float(np.percentile(clean_deltas, 95)), 5)))
    print("  max:   " + str(round(float(clean_deltas.max()), 5)))
    print("  min:   " + str(round(float(clean_deltas.min()), 5)))

    # GQA hypothesis: do retrieval heads cluster by KV group?
    print("\n=== GQA Hypothesis: KV Group Distribution of Retrieval Heads ===")
    retrieval_kv_groups = [results[str(l) + "_" + str(h)]["kv_group"]
                            for d, l, h, kv_grp in all_deltas
                            if labels.get(str(l) + "_" + str(h)) == "retrieval"]
    if retrieval_kv_groups:
        from collections import Counter
        grp_counts = Counter(retrieval_kv_groups)
        print("  KV group 0 (Q heads 0-6):  " + str(grp_counts.get(0, 0)) + " retrieval heads")
        print("  KV group 1 (Q heads 7-13): " + str(grp_counts.get(1, 0)) + " retrieval heads")
        print("  (Clustered = GQA retrieval hypothesis CONFIRMED)")
        print("  (Even split = KV group doesn't predict retrieval specialization)")
    else:
        print("  No retrieval heads found above threshold " + str(THRESHOLD_RETRIEVAL))
        print("  -> Qwen-0.5B may not develop distinct retrieval heads under GQA")

    # Spatial law check for Qwen
    print("\n=== Spatial Law: Mean Depth per Role ===")
    role_depths = {"sink": [], "retrieval": [], "induction": [], "local": []}
    for key, role in labels.items():
        l, h = map(int, key.split("_"))
        rel_depth = l / (num_layers - 1)
        role_depths[role].append(rel_depth)
    for role in ["sink", "retrieval", "induction", "local"]:
        depths = role_depths[role]
        if not depths:
            print("  " + role + ": no heads")
            continue
        d = np.array(depths)
        print("  " + role.ljust(12) +
              " mean=" + str(round(float(d.mean()), 3)) +
              " std="  + str(round(float(d.std()), 3)) +
              " n="    + str(len(depths)))

    if nan_heads:
        print("\n  NaN/sink heads: " + str(len(nan_heads)))
        for l, h, kv in nan_heads:
            print("    layer=" + str(l) + " head=" + str(h) + " kv_group=" + str(kv))

    # Save
    out = {
        "model":      MODEL_ID,
        "n_pairs":    len(PROMPT_PAIRS),
        "architecture": {
            "num_layers":    num_layers,
            "num_q_heads":   num_q_heads,
            "num_kv_heads":  num_kv_heads,
            "gqa_group_size": gqa_group,
        },
        "thresholds": {
            "retrieval_delta_min": THRESHOLD_RETRIEVAL,
            "induction_delta_max": THRESHOLD_INDUCTION,
            "sink_entropy_max":    THRESHOLD_SINK_ENT,
        },
        "counts": counts,
        "labels": labels,
        "heads":  results,
    }
    out_path = os.path.join(OUT_DIR, "qwen0.5b_retrieval_entropy.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> " + out_path)
    print("[DONE]")


if __name__ == "__main__":
    main()
