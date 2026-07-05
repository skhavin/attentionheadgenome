"""
step0_extract_dataset.py
------------------------
Extract 100 prompts from two sources and save as shared JSON cache:
  1. WikiText-103-v1 validation split (natural English, long-form)
  2. Universal Dependencies English EWT treebank (grammar-labelled sentences)

Run this once. All other scripts read from outputs/phase2_atlas/dataset.json.

Output schema:
{
  "wikitext": [{"text": str}, ...],          # 100 entries
  "ud_ewt":   [{"text": str, "tokens": [...], "deps": [...], "pos": [...]}, ...]  # 100 entries
}
"""

import json, os
from datasets import load_dataset

OUT_DIR = "outputs/phase2_atlas"
OUT_FILE = os.path.join(OUT_DIR, "dataset.json")
N = 100
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. WikiText-103 ──────────────────────────────────────────────────────────
print("Loading WikiText-103 validation split...")
wt = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")

wikitext_samples = []
for row in wt:
    text = row["text"].strip()
    # Skip section headers and empty lines
    if len(text) > 60 and not text.startswith("="):
        wikitext_samples.append({"text": text})
    if len(wikitext_samples) >= N:
        break

print(f"  Collected {len(wikitext_samples)} WikiText sentences.")

# ── 2. Universal Dependencies English EWT ────────────────────────────────────
print("Loading UD English EWT...")
# 'universal_dependencies' dataset with 'en_ewt' config
ud = load_dataset("universal-dependencies/universal_dependencies", "en_ewt", split="dev", trust_remote_code=True)

ud_samples = []
for row in ud:
    tokens = row["tokens"]
    deprels = row["deprel"]
    pos_tags = row["upos"]

    # Build plain text from tokens
    text = " ".join(tokens)
    if len(text) < 20:
        continue

    # Map integer POS to string using the dataset's feature
    ud_samples.append({
        "text": text,
        "tokens": tokens,
        "deps": deprels,
        "pos": pos_tags
    })
    if len(ud_samples) >= N:
        break

print(f"  Collected {len(ud_samples)} UD-EWT sentences.")

# ── Save ─────────────────────────────────────────────────────────────────────
dataset = {"wikitext": wikitext_samples, "ud_ewt": ud_samples}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nSaved dataset to {OUT_FILE}")
print(f"  WikiText: {len(wikitext_samples)} samples")
print(f"  UD-EWT:   {len(ud_samples)} samples")
