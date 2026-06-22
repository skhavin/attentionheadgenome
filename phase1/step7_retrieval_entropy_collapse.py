# phase1/step7_retrieval_entropy_collapse.py
#
# PURPOSE:
#   Empirically validate whether any attention heads exhibit "retrieval behavior":
#   entropy collapse when a matching key appears in context (vs. absent).
#
# EXPERIMENT DESIGN (mechanistically grounded):
#   For each head, compute:
#     entropy_match    = mean attention entropy on MATCHING prompts
#     entropy_nonmatch = mean attention entropy on NON-MATCHING prompts
#     delta            = entropy_nonmatch - entropy_match
#
#   True retrieval head: large positive delta (collapses onto match, high entropy without it)
#   Sink/local/background head: near-zero delta (pattern doesn't change with match presence)
#
# SCOPE: GPT-2 Medium only (silhouette=0.4679, strongest clustering, 384 heads)
#        Run on Qwen/Llama only if GPT-2 shows clear retrieval heads.
#
# OUTPUTS:
#   outputs/phase1/gpt2_retrieval_entropy.json

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "phase1")

MODEL_ID = "gpt2-medium"

# ---------------------------------------------------------------------------
# Synthetic prompt pairs
# Each pair: (matching_context, nonmatching_context, query_suffix)
# The query asks for a fact that IS present in match but ABSENT in nonmatch.
# ---------------------------------------------------------------------------
PROMPT_PAIRS = [
    (
        "The capital of France is Paris.",
        "The weather today is sunny and warm.",
        " The capital of France is",
    ),
    (
        "The speed of light is 299792458 meters per second.",
        "The dog ran quickly across the field.",
        " The speed of light is",
    ),
    (
        "Shakespeare was born in Stratford-upon-Avon.",
        "The stock market closed higher yesterday.",
        " Shakespeare was born in",
    ),
    (
        "Water boils at 100 degrees Celsius.",
        "The train arrived at the station late.",
        " Water boils at",
    ),
    (
        "The Eiffel Tower is located in Paris.",
        "She enjoyed reading books in the evening.",
        " The Eiffel Tower is located in",
    ),
    (
        "Mount Everest is the tallest mountain on Earth.",
        "He enjoyed cooking pasta for dinner.",
        " Mount Everest is the tallest",
    ),
    (
        "The chemical formula for water is H2O.",
        "The children played outside all afternoon.",
        " The chemical formula for water is",
    ),
    (
        "Leonardo da Vinci painted the Mona Lisa.",
        "The conference was held in a large hall.",
        " Leonardo da Vinci painted the",
    ),
    (
        "The Great Wall of China was built over many centuries.",
        "The new restaurant opened last week downtown.",
        " The Great Wall of China was",
    ),
    (
        "Albert Einstein developed the theory of relativity.",
        "The supermarket was crowded on Saturday morning.",
        " Albert Einstein developed the theory of",
    ),
    (
        "Oxygen has the atomic number 8.",
        "The library closed early due to renovations.",
        " Oxygen has the atomic number",
    ),
    (
        "The Amazon River flows through Brazil.",
        "She spent the afternoon painting in her studio.",
        " The Amazon River flows through",
    ),
    (
        "The human body has 206 bones.",
        "The cat curled up on the warm windowsill.",
        " The human body has",
    ),
    (
        "Isaac Newton discovered gravity.",
        "They drove through the mountains on their vacation.",
        " Isaac Newton discovered",
    ),
    (
        "The Pacific Ocean is the largest ocean on Earth.",
        "He forgot to bring his umbrella to work.",
        " The Pacific Ocean is the",
    ),
    (
        "DNA stands for deoxyribonucleic acid.",
        "The children built a sandcastle on the beach.",
        " DNA stands for",
    ),
    (
        "Rome is the capital of Italy.",
        "She planted roses in her garden this spring.",
        " Rome is the capital of",
    ),
    (
        "The Berlin Wall fell in 1989.",
        "He enjoyed hiking in the national park.",
        " The Berlin Wall fell in",
    ),
    (
        "The Pythagorean theorem states that a squared plus b squared equals c squared.",
        "She read a mystery novel before going to sleep.",
        " The Pythagorean theorem states",
    ),
    (
        "Photosynthesis converts sunlight into chemical energy.",
        "They watched the fireworks from the hilltop.",
        " Photosynthesis converts sunlight into",
    ),
]


def attention_entropy(attn_weights):
    """
    Compute mean entropy of attention distribution across heads.
    attn_weights: (num_heads, seq_len, seq_len)
    Returns: (num_heads,) entropy values
    """
    # Only use the last query position (token before the answer begins)
    last_pos = attn_weights[:, -1, :]  # (num_heads, seq_len)
    p = last_pos + 1e-12
    p = p / p.sum(dim=-1, keepdim=True)  # renormalise (should already sum to 1)
    entropy = -torch.sum(p * torch.log(p), dim=-1)  # (num_heads,)
    return entropy.cpu().numpy()


def run_prompt(model, tokenizer, text, device):
    """
    Run a single text string through GPT-2 and return per-layer per-head entropy
    at the last token position.
    Returns: list of (num_heads,) arrays, one per layer.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    # out.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    # NOTE: requires attn_implementation='eager'; SDPA/flash returns None here.
    layer_entropies = []
    for attn in out.attentions:
        if attn is None:
            raise RuntimeError(
                "Attention weights are None. Model must be loaded with "
                "attn_implementation='eager'."
            )
        layer_entropies.append(attention_entropy(attn[0]))  # remove batch dim
    return layer_entropies  # list[layer] of (num_heads,)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    print("Loading GPT-2 Medium (attn_implementation=eager to get attention weights)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
    )
    if device == "cuda":
        model = model.half().cuda()
    model.eval()

    num_layers = model.config.n_layer   # 24
    num_heads  = model.config.n_head    # 16

    # Accumulate entropy deltas per (layer, head)
    match_entropies    = [[[] for _ in range(num_heads)] for _ in range(num_layers)]
    nonmatch_entropies = [[[] for _ in range(num_heads)] for _ in range(num_layers)]

    print("Running " + str(len(PROMPT_PAIRS)) + " synthetic prompt pairs...")
    for i, (match_ctx, nonmatch_ctx, query) in enumerate(PROMPT_PAIRS):
        match_text    = match_ctx    + query
        nonmatch_text = nonmatch_ctx + query

        match_ent    = run_prompt(model, tokenizer, match_text,    device)
        nonmatch_ent = run_prompt(model, tokenizer, nonmatch_text, device)

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                match_entropies[layer_idx][head_idx].append(
                    float(match_ent[layer_idx][head_idx])
                )
                nonmatch_entropies[layer_idx][head_idx].append(
                    float(nonmatch_ent[layer_idx][head_idx])
                )

        if (i + 1) % 5 == 0:
            print("  Processed " + str(i + 1) + "/" + str(len(PROMPT_PAIRS)) + " pairs")

    # Compute per-head delta = mean(nonmatch_entropy) - mean(match_entropy)
    print("\nComputing entropy-collapse deltas...")
    results = {}
    all_deltas = []
    nan_heads = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            me  = float(np.mean(match_entropies[layer_idx][head_idx]))
            nme = float(np.mean(nonmatch_entropies[layer_idx][head_idx]))
            delta = nme - me  # positive = collapsed on match, stayed high on nonmatch
            key = str(layer_idx) + "_" + str(head_idx)

            if np.isnan(delta) or np.isnan(me) or np.isnan(nme):
                nan_heads.append((layer_idx, head_idx))
                results[key] = {
                    "match_entropy":    None,
                    "nonmatch_entropy": None,
                    "delta":            None,
                    "nan": True,
                }
                # Exclude NaN heads from ranking; use 0.0 as a sentinel
                delta = 0.0
                me, nme = 0.0, 0.0
            else:
                results[key] = {
                    "match_entropy":    round(me, 5),
                    "nonmatch_entropy": round(nme, 5),
                    "delta":            round(delta, 5),
                }
            all_deltas.append((delta, layer_idx, head_idx))

    if nan_heads:
        print("  WARNING: " + str(len(nan_heads)) + " heads had NaN entropy (likely from zero attention rows).")
        for l, h in nan_heads:
            print("    NaN head: layer=" + str(l) + " head=" + str(h))


    # Sort by delta descending — top are retrieval candidates
    all_deltas.sort(reverse=True)

    print("\n--- Top 20 Entropy-Collapse Heads (retrieval candidates) ---")
    print("  rank  layer  head   delta   match_ent  nonmatch_ent")
    for rank, (delta, layer_idx, head_idx) in enumerate(all_deltas[:20]):
        key = str(layer_idx) + "_" + str(head_idx)
        me  = results[key]["match_entropy"]
        nme = results[key]["nonmatch_entropy"]
        print(
            "  " + str(rank + 1).ljust(5) +
            str(layer_idx).ljust(7) +
            str(head_idx).ljust(7) +
            str(round(delta, 4)).ljust(9) +
            str(round(me, 4)).ljust(12) +
            str(round(nme, 4))
        )

    print("\n--- Bottom 10 heads (no retrieval behavior) ---")
    print("  rank  layer  head   delta")
    for rank, (delta, layer_idx, head_idx) in enumerate(all_deltas[-10:]):
        actual_rank = len(all_deltas) - 10 + rank + 1
        print(
            "  " + str(actual_rank).ljust(5) +
            str(layer_idx).ljust(7) +
            str(head_idx).ljust(7) +
            str(round(delta, 4))
        )

    # Distribution statistics
    deltas_arr = np.array([d for d, _, _ in all_deltas])
    print("\n--- Delta Distribution ---")
    print("  mean:   " + str(round(float(deltas_arr.mean()), 5)))
    print("  std:    " + str(round(float(deltas_arr.std()), 5)))
    print("  p75:    " + str(round(float(np.percentile(deltas_arr, 75)), 5)))
    print("  p90:    " + str(round(float(np.percentile(deltas_arr, 90)), 5)))
    print("  p95:    " + str(round(float(np.percentile(deltas_arr, 95)), 5)))
    print("  p99:    " + str(round(float(np.percentile(deltas_arr, 99)), 5)))
    print("  max:    " + str(round(float(deltas_arr.max()), 5)))
    print("  min:    " + str(round(float(deltas_arr.min()), 5)))

    # Save
    out = {
        "model":  MODEL_ID,
        "n_pairs": len(PROMPT_PAIRS),
        "delta_stats": {
            "mean": round(float(deltas_arr.mean()), 5),
            "std":  round(float(deltas_arr.std()), 5),
            "p75":  round(float(np.percentile(deltas_arr, 75)), 5),
            "p90":  round(float(np.percentile(deltas_arr, 90)), 5),
            "p95":  round(float(np.percentile(deltas_arr, 95)), 5),
            "max":  round(float(deltas_arr.max()), 5),
            "min":  round(float(deltas_arr.min()), 5),
        },
        "heads": results,
    }
    out_path = os.path.join(OUT_DIR, "gpt2_retrieval_entropy.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved -> " + out_path)
    print("[DONE]")


if __name__ == "__main__":
    main()
