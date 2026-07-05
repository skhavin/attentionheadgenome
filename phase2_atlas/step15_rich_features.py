"""
step15_rich_features.py  (Workstream 1 - Feature Collection)
-------------------------------------------------------------
Extends the existing atlas data with runtime behavioral features
not captured in the original step1-step5 pipeline:

  - position_bias_profile  : attention mass distribution across thirds of context
  - activation_sparsity    : fraction of tokens this head attends to meaningfully (>0.05 mass)
  - inter_layer_correlation: how correlated this head's attention map is with same-index head in adjacent layers

Runs on Qwen2.5-0.5B, Qwen2.5-1.5B, GPT-2-medium, Llama-3.2-1B.
Output: outputs/routing/{MODEL}_rich_features.json
"""
import json, os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODELS = [
    "gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "unsloth/Llama-3.2-1B",
]

N_PROMPTS = 50

with open("outputs/phase2_atlas/dataset.json") as f:
    dataset = json.load(f)
# Mix: 25 wikitext + 25 ud_ewt for linguistic diversity
texts = (
    [s["text"] for s in dataset["wikitext"][:25]] +
    [" ".join(s["tokens"]) for s in dataset["ud_ewt"][:25]]
)

os.makedirs("outputs/routing", exist_ok=True)

for MODEL_NAME in MODELS:
    SAFE = MODEL_NAME.split("/")[-1]
    out_path = f"outputs/routing/{SAFE}_rich_features.json"
    if os.path.exists(out_path):
        print(f"\n{SAFE}: already done, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"MODEL: {SAFE}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    model.eval()

    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads

    # Accumulators per head
    pos_bias_early  = {(l,h): [] for l in range(L) for h in range(H)}  # first third
    pos_bias_middle = {(l,h): [] for l in range(L) for h in range(H)}
    pos_bias_late   = {(l,h): [] for l in range(L) for h in range(H)}  # last third
    sparsity_acc    = {(l,h): [] for l in range(L) for h in range(H)}
    attn_maps       = {(l,h): [] for l in range(L) for h in range(H)}  # for inter-layer corr

    valid = 0
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        T = ids["input_ids"].shape[1]
        if T < 12:
            continue

        with torch.no_grad():
            out = model(**ids, output_attentions=True)

        third = T // 3
        for l, attn in enumerate(out.attentions):
            a = attn[0].float().cpu().numpy()  # (H, T, T)
            for h in range(H):
                # Use mean attention row across all query positions
                row = a[h].mean(axis=0)  # (T,) — avg attention received per key position

                # Position bias profile
                pos_bias_early[(l,h)].append(float(row[:third].sum()))
                pos_bias_middle[(l,h)].append(float(row[third:2*third].sum()))
                pos_bias_late[(l,h)].append(float(row[2*third:].sum()))

                # Activation sparsity: fraction of keys receiving > 5% attention
                sparsity_acc[(l,h)].append(float((row > 0.05).mean()))

                # Store a compact attention map signature (last row only) for correlation
                attn_maps[(l,h)].append(a[h, -1, :])

        valid += 1

    print(f"  Processed {valid} prompts.")

    # Compute inter-layer correlation (between layer l and l+1 for same head)
    results = {}
    for l in range(L):
        for h in range(H):
            key = f"{l}_{h}"
            early_m  = float(np.mean(pos_bias_early[(l,h)]))  if pos_bias_early[(l,h)]  else 0.0
            middle_m = float(np.mean(pos_bias_middle[(l,h)])) if pos_bias_middle[(l,h)] else 0.0
            late_m   = float(np.mean(pos_bias_late[(l,h)]))   if pos_bias_late[(l,h)]   else 0.0
            spars    = float(np.mean(sparsity_acc[(l,h)]))     if sparsity_acc[(l,h)]    else 0.0

            # Inter-layer correlation: correlate last-row attn vectors with adjacent layer
            # We align samples across layers (same prompt gives same-length rows)
            inter_corr = None
            if l < L - 1 and attn_maps[(l,h)] and attn_maps[(l+1,h)]:
                pairs = list(zip(attn_maps[(l,h)], attn_maps[(l+1,h)]))
                # Pad/trim to same length per prompt
                corrs = []
                for a1, a2 in pairs:
                    min_len = min(len(a1), len(a2))
                    if min_len > 2:
                        corrs.append(float(np.corrcoef(a1[:min_len], a2[:min_len])[0, 1]))
                inter_corr = float(np.mean(corrs)) if corrs else None

            results[key] = {
                "position_bias": {
                    "early_third":  round(early_m,  4),
                    "middle_third": round(middle_m, 4),
                    "late_third":   round(late_m,   4),
                },
                "activation_sparsity": round(spars, 4),
                "inter_layer_corr":    round(inter_corr, 4) if inter_corr is not None else None,
            }

    with open(out_path, "w") as f:
        json.dump({"model": SAFE, "heads": results}, f, indent=2)
    print(f"  Saved to {out_path}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nAll models done.")
