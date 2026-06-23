# -*- coding: utf-8 -*-
# phase1/step12_robust_entropy_50pairs.py
#
# PURPOSE:
#   Address Gap 1 (n=20 is thin) and Gap 3 (threshold hand-tuned).
#
#   1. Expand to 50 diverse prompt pairs covering 5 semantic categories:
#      geography, science, history, literature, mathematics
#   2. Run across GPT-2-Medium, Qwen-2.5-0.5B, Qwen-2.5-1.5B
#   3. Perform threshold sensitivity analysis: count retrieval/induction heads
#      at delta thresholds from 0.15 to 0.45 in steps of 0.05
#
# OUTPUTS:
#   outputs/phase1/robust_entropy_{model}.json       -- raw per-head deltas (50 pairs)
#   outputs/phase1/threshold_sensitivity.json        -- sensitivity table across thresholds

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")

MODELS = [
    {
        "model_id":   "gpt2-medium",
        "slug":       "gpt2",
        "dtype":      "float32",
        "num_layers": 24,
        "num_q":      16,
        "num_kv":     16,
        "gqa_group":  1,
    },
    {
        "model_id":   "Qwen/Qwen2.5-0.5B",
        "slug":       "qwen0.5b",
        "dtype":      "bfloat16",
        "num_layers": 24,
        "num_q":      14,
        "num_kv":     2,
        "gqa_group":  7,
    },
    {
        "model_id":   "Qwen/Qwen2.5-1.5B",
        "slug":       "qwen1.5b",
        "dtype":      "bfloat16",
        "num_layers": 28,
        "num_q":      12,
        "num_kv":     2,
        "gqa_group":  6,
    },
]

THRESHOLD_RETRIEVAL = 0.30
THRESHOLD_INDUCTION = -0.50

# 50 prompt pairs across 5 categories (10 each)
# Format: (match_context, nonmatch_context, query_suffix)
PROMPT_PAIRS = [
    # -- Geography (10) --
    ("The capital of France is Paris.",
     "The weather today is sunny and warm.",
     " The capital of France is"),
    ("The Nile is the longest river in the world.",
     "She enjoyed reading books in the evening.",
     " The Nile is the longest"),
    ("The Eiffel Tower is located in Paris.",
     "The stock market closed higher yesterday.",
     " The Eiffel Tower is located in"),
    ("Mount Everest is the tallest mountain on Earth.",
     "He enjoyed cooking pasta for dinner.",
     " Mount Everest is the tallest"),
    ("The Pacific Ocean is the largest ocean on Earth.",
     "He forgot to bring his umbrella to work.",
     " The Pacific Ocean is the"),
    ("The Amazon River flows through Brazil.",
     "She spent the afternoon painting in her studio.",
     " The Amazon River flows through"),
    ("Rome is the capital of Italy.",
     "She planted roses in her garden this spring.",
     " Rome is the capital of"),
    ("The Sahara is the largest hot desert in the world.",
     "The train arrived at the station late.",
     " The Sahara is the largest"),
    ("Japan is an island nation in East Asia.",
     "The children played outside all afternoon.",
     " Japan is an island nation in"),
    ("The Great Wall of China stretches over 13000 miles.",
     "The conference was held in a large hall.",
     " The Great Wall of China stretches over"),
    # -- Science (10) --
    ("The speed of light is 299792458 meters per second.",
     "The dog ran quickly across the field.",
     " The speed of light is"),
    ("Water boils at 100 degrees Celsius at sea level.",
     "The train arrived at the station late.",
     " Water boils at"),
    ("The chemical formula for water is H2O.",
     "The children played outside all afternoon.",
     " The chemical formula for water is"),
    ("Oxygen has the atomic number 8.",
     "The library closed early due to renovations.",
     " Oxygen has the atomic number"),
    ("DNA stands for deoxyribonucleic acid.",
     "The children built a sandcastle on the beach.",
     " DNA stands for"),
    ("Photosynthesis converts sunlight into chemical energy.",
     "They watched the fireworks from the hilltop.",
     " Photosynthesis converts sunlight into"),
    ("The human body has 206 bones.",
     "The cat curled up on the warm windowsill.",
     " The human body has"),
    ("The Earth orbits the Sun once every 365 days.",
     "He baked a cake for the birthday party.",
     " The Earth orbits the Sun once every"),
    ("Gravity was described by Newton as an attractive force between masses.",
     "She reorganized the books on the shelf.",
     " Gravity was described by Newton as"),
    ("Light travels faster than sound in air.",
     "They adopted a puppy from the shelter.",
     " Light travels faster than"),
    # -- History (10) --
    ("The Berlin Wall fell in 1989.",
     "He enjoyed hiking in the national park.",
     " The Berlin Wall fell in"),
    ("Shakespeare was born in Stratford-upon-Avon.",
     "The supermarket was crowded on Saturday morning.",
     " Shakespeare was born in"),
    ("The French Revolution began in 1789.",
     "She organized her wardrobe on Sunday afternoon.",
     " The French Revolution began in"),
    ("World War II ended in 1945.",
     "He went jogging along the river each morning.",
     " World War II ended in"),
    ("The Roman Empire fell in 476 AD.",
     "She prepared a fresh salad for dinner.",
     " The Roman Empire fell in"),
    ("Christopher Columbus arrived in the Americas in 1492.",
     "The dog learned a new trick in the park.",
     " Christopher Columbus arrived in the Americas in"),
    ("The Declaration of Independence was signed in 1776.",
     "He repaired the bicycle in the garage.",
     " The Declaration of Independence was signed in"),
    ("The Moon landing happened in July 1969.",
     "She watered the plants on the balcony.",
     " The Moon landing happened in"),
    ("Napoleon was exiled to the island of Elba.",
     "The library had a new collection of novels.",
     " Napoleon was exiled to"),
    ("The Treaty of Versailles was signed in 1919.",
     "The children decorated the classroom with drawings.",
     " The Treaty of Versailles was signed in"),
    # -- Literature (10) --
    ("Leonardo da Vinci painted the Mona Lisa.",
     "The new restaurant opened last week downtown.",
     " Leonardo da Vinci painted the"),
    ("Hamlet is a play written by Shakespeare.",
     "The supermarket had a sale on fresh produce.",
     " Hamlet is a play written by"),
    ("George Orwell wrote the novel 1984.",
     "She arranged fresh flowers in a vase.",
     " George Orwell wrote the novel"),
    ("The Great Gatsby was written by F. Scott Fitzgerald.",
     "He mended the fence in the backyard.",
     " The Great Gatsby was written by"),
    ("Don Quixote was written by Miguel de Cervantes.",
     "The children built a fort out of pillows.",
     " Don Quixote was written by"),
    ("Homer wrote the Iliad and the Odyssey.",
     "The museum displayed a new art collection.",
     " Homer wrote the"),
    ("Tolstoy wrote War and Peace.",
     "The bakery sold fresh bread every morning.",
     " Tolstoy wrote"),
    ("Mary Shelley wrote Frankenstein.",
     "The park was crowded with families on Sunday.",
     " Mary Shelley wrote"),
    ("Charles Dickens wrote A Tale of Two Cities.",
     "She fixed the leaking pipe in the kitchen.",
     " Charles Dickens wrote"),
    ("Dante wrote the Divine Comedy.",
     "They rearranged the furniture in the living room.",
     " Dante wrote"),
    # -- Mathematics (10) --
    ("The Pythagorean theorem states that a squared plus b squared equals c squared.",
     "She read a mystery novel before going to sleep.",
     " The Pythagorean theorem states"),
    ("Pi is approximately equal to 3.14159.",
     "He took the dog for a walk in the park.",
     " Pi is approximately equal to"),
    ("The square root of 144 is 12.",
     "She watered the plants and trimmed the hedges.",
     " The square root of 144 is"),
    ("Euler's number e is approximately 2.71828.",
     "He painted the fence white over the weekend.",
     " Euler's number e is approximately"),
    ("A prime number has no divisors other than 1 and itself.",
     "The children played board games on rainy days.",
     " A prime number has no divisors other than"),
    ("The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5.",
     "She made lemonade for the summer fair.",
     " The Fibonacci sequence starts with"),
    ("The sum of angles in a triangle is 180 degrees.",
     "He organized his stamp collection by country.",
     " The sum of angles in a triangle is"),
    ("A circle's circumference equals pi times the diameter.",
     "She learned to knit from her grandmother.",
     " A circle's circumference equals"),
    ("The derivative of x squared is 2x.",
     "He assembled a model airplane kit.",
     " The derivative of x squared is"),
    ("The quadratic formula solves ax squared plus bx plus c equals zero.",
     "She attended a pottery class on Saturdays.",
     " The quadratic formula solves"),
]

THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def attention_entropy(attn_weights):
    last_pos = attn_weights[:, -1, :].float()
    nan_mask = torch.isnan(last_pos)
    last_pos = torch.where(nan_mask, torch.zeros_like(last_pos), last_pos)
    row_sum = last_pos.sum(dim=-1, keepdim=True)
    zero_rows = (row_sum == 0).squeeze(-1)
    if zero_rows.any():
        last_pos[zero_rows] = 1.0 / last_pos.shape[-1]
        row_sum = last_pos.sum(dim=-1, keepdim=True)
    p = last_pos / row_sum + 1e-12
    p = p / p.sum(dim=-1, keepdim=True)
    return (-torch.sum(p * torch.log(p), dim=-1)).cpu().numpy()


def run_prompt(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    return [attention_entropy(a[0]) for a in out.attentions]


def run_model(cfg, device):
    slug = cfg["slug"]
    print("\n" + "="*60)
    print(f"Running: {cfg['model_id']}")
    print("="*60)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[cfg["dtype"]]

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        attn_implementation="eager",
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    L = cfg["num_layers"]
    H = cfg["num_q"]
    G = cfg["gqa_group"]

    match_ent    = [[[] for _ in range(H)] for _ in range(L)]
    nonmatch_ent = [[[] for _ in range(H)] for _ in range(L)]

    print(f"  Running {len(PROMPT_PAIRS)} prompt pairs...")
    for i, (ctx_m, ctx_nm, query) in enumerate(PROMPT_PAIRS):
        me  = run_prompt(model, tok, ctx_m  + query, device)
        nme = run_prompt(model, tok, ctx_nm + query, device)
        for l in range(L):
            for h in range(H):
                match_ent[l][h].append(float(me[l][h]))
                nonmatch_ent[l][h].append(float(nme[l][h]))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(PROMPT_PAIRS)} pairs done")

    # Compute per-head deltas
    results = {}
    all_deltas = []
    nan_count = 0

    for l in range(L):
        for h in range(H):
            me_val  = float(np.mean(match_ent[l][h]))
            nme_val = float(np.mean(nonmatch_ent[l][h]))
            delta   = nme_val - me_val
            key     = f"{l}_{h}"
            kv_grp  = h // G

            if np.isnan(delta) or np.isnan(me_val):
                nan_count += 1
                results[key] = {"delta": None, "match_entropy": None,
                                "nonmatch_entropy": None, "kv_group": kv_grp, "nan": True}
                all_deltas.append((0.0, l, h))
            else:
                results[key] = {
                    "delta":            round(delta, 5),
                    "match_entropy":    round(me_val, 5),
                    "nonmatch_entropy": round(nme_val, 5),
                    "kv_group":         kv_grp,
                }
                all_deltas.append((delta, l, h))

    del model  # free GPU memory before next model
    torch.cuda.empty_cache()

    # --- Threshold sensitivity ---
    clean_deltas = np.array([d for d, _, _ in all_deltas if not np.isnan(d)])
    sensitivity = {}
    for thr in THRESHOLDS:
        n_ret = int((clean_deltas > thr).sum())
        n_ind = int((clean_deltas < -thr * 1.667).sum())  # keep same ratio as original
        sensitivity[str(thr)] = {"retrieval": n_ret, "induction": n_ind}

    print(f"\n  NaN heads: {nan_count}")
    print(f"\n  Threshold sensitivity (retrieval count | induction count):")
    print(f"  {'Delta':>8}  {'Retrieval':>10}  {'Induction':>10}")
    for thr in THRESHOLDS:
        r = sensitivity[str(thr)]["retrieval"]
        i = sensitivity[str(thr)]["induction"]
        marker = " <-- baseline" if thr == THRESHOLD_RETRIEVAL else ""
        print(f"  {thr:>8.2f}  {r:>10}  {i:>10}{marker}")

    out = {
        "model": cfg["model_id"],
        "n_pairs": len(PROMPT_PAIRS),
        "architecture": {"L": L, "H": H, "kv": cfg["num_kv"], "gqa_group": G},
        "thresholds": {"retrieval": THRESHOLD_RETRIEVAL, "induction": THRESHOLD_INDUCTION},
        "heads": results,
        "threshold_sensitivity": sensitivity,
        "delta_stats": {
            "mean":  round(float(clean_deltas.mean()), 5),
            "std":   round(float(clean_deltas.std()), 5),
            "p90":   round(float(np.percentile(clean_deltas, 90)), 5),
            "p95":   round(float(np.percentile(clean_deltas, 95)), 5),
            "max":   round(float(clean_deltas.max()), 5),
            "min":   round(float(clean_deltas.min()), 5),
        }
    }
    out_path = os.path.join(OUT_DIR, f"robust_entropy_{slug}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    return slug, sensitivity


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_sensitivity = {}
    for cfg in MODELS:
        slug, sens = run_model(cfg, device)
        all_sensitivity[slug] = sens

    # Cross-model sensitivity summary
    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY: Retrieval Head Count Across Models & Thresholds")
    print("="*70)
    print(f"  {'Threshold':<12}", end="")
    for cfg in MODELS:
        print(f"  {cfg['slug']:>12}", end="")
    print()
    for thr in THRESHOLDS:
        marker = " <--" if thr == THRESHOLD_RETRIEVAL else "    "
        print(f"  {thr:<12.2f}", end="")
        for cfg in MODELS:
            n = all_sensitivity[cfg["slug"]][str(thr)]["retrieval"]
            print(f"  {n:>12}", end="")
        print(marker)

    # Save combined sensitivity table
    combined_path = os.path.join(OUT_DIR, "threshold_sensitivity.json")
    with open(combined_path, "w") as f:
        json.dump({
            "thresholds": THRESHOLDS,
            "models": all_sensitivity
        }, f, indent=2)
    print(f"\nCombined sensitivity saved -> {combined_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
