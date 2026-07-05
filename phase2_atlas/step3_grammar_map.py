"""
step3_grammar_map.py  (Pillar 3 — UD Grammar Mapping)
------------------------------------------------------
Using the 100 Universal Dependencies sentences from dataset.json,
for each head measure what fraction of its attention mass lands on
tokens of each dependency role (nsubj, obj, amod, root, det, advmod, case, punct, other).

This reveals whether a head systematically attends to grammatical subjects,
verbs, objects, determiners, etc.

Output: outputs/phase2_atlas/grammar_map.json
Schema: {
  "model": str,
  "dep_labels": [list],
  "heads": { "L_H": { "nsubj": float, "obj": float, ... } }
}
"""

import json, os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
SAFE_MODEL = MODEL.split("/")[-1]
DATASET = "outputs/phase2_atlas/dataset.json"
OUT     = f"outputs/phase2_atlas/{SAFE_MODEL}_grammar_map.json"

# The dependency labels we track explicitly
LABELS_OF_INTEREST = ["nsubj", "obj", "amod", "root", "det", "advmod", "case", "punct"]

device = "cuda" if torch.cuda.is_available() else "cpu"
tok    = AutoTokenizer.from_pretrained(MODEL)
model  = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

with open(DATASET) as f:
    data = json.load(f)

ud_samples = data["ud_ewt"]
L = model.config.num_hidden_layers
H = model.config.num_attention_heads

# Accumulators: per (layer, head), per dep label, accumulated attention mass
# We use the last-token attention row as the probe
dep_acc    = {(l, h): {lbl: [] for lbl in LABELS_OF_INTEREST + ["other"]}
              for l in range(L) for h in range(H)}
n_processed = 0

print(f"Running {len(ud_samples)} UD sentences...")
for i, sample in enumerate(ud_samples):
    word_tokens = sample["tokens"]   # list of str
    dep_roles   = sample["deps"]     # list of str (one per word token)

    text = " ".join(word_tokens)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    subword_ids = ids["input_ids"][0]
    T = len(subword_ids)

    if T < 3:
        continue

    # Map each subword token index to the word-level dep label
    # We do this by comparing subword tokenization offsets with word boundaries
    word_to_dep = {}  # word_idx -> dep label
    for wi, dep in enumerate(dep_roles):
        word_to_dep[wi] = dep

    # Greedily align subwords → words (approximate: re-tokenize each word)
    subword_to_dep = ["other"] * T
    word_idx = 0
    # Encode each word separately and count its subwords
    pos = 0  # current subword position (skip BOS if any)
    # GPT-2 tokenizer adds no explicit BOS, but encoding full text vs pieces may differ.
    # Simple approach: encode word by word and assign dep label to each produced token.
    for wi, word in enumerate(word_tokens):
        dep_label = dep_roles[wi] if wi < len(dep_roles) else "other"
        if dep_label not in LABELS_OF_INTEREST:
            dep_label = "other"
        sub = tok(" " + word, add_special_tokens=False)["input_ids"]
        for _ in sub:
            if pos < T:
                subword_to_dep[pos] = dep_label
                pos += 1

    with torch.no_grad():
        out = model(**ids, output_attentions=True)

    # Use last token's attention row
    for l, attn in enumerate(out.attentions):
        a = attn[0].float().cpu().numpy()  # (H, T, T)
        for h in range(H):
            row = a[h, -1, :]  # attention from last token to all others

            # Accumulate mass by dep label
            mass_per_label = {lbl: 0.0 for lbl in LABELS_OF_INTEREST + ["other"]}
            for t_idx, dep in enumerate(subword_to_dep):
                if dep in mass_per_label:
                    mass_per_label[dep] += float(row[t_idx])
                else:
                    mass_per_label["other"] += float(row[t_idx])

            for lbl in LABELS_OF_INTEREST + ["other"]:
                dep_acc[(l, h)][lbl].append(mass_per_label[lbl])

    n_processed += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(ud_samples)} done")

print(f"  Processed {n_processed} sentences.")

# Aggregate
results = {}
for (l, h), label_dict in dep_acc.items():
    results[f"{l}_{h}"] = {
        lbl: round(float(np.mean(vals)), 4) if vals else 0.0
        for lbl, vals in label_dict.items()
    }

out_data = {
    "model":      MODEL,
    "dep_labels": LABELS_OF_INTEREST + ["other"],
    "heads":      results,
}
with open(OUT, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved to {OUT}")
