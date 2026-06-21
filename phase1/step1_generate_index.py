# step1_generate_index.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Lock down 300 shared WikiText-103 article indices.
#          Every profiling script reads this file — no model ever sees different documents.
#
# OUTPUTS:
#   outputs/phase1/dataset_index.json
#
# HOW IT WORKS:
#   1. Load all WikiText-103 train rows (same logic as old-proj/data_utils.py).
#   2. Group rows into full articles (split on "= Title =" headings).
#   3. Use random.seed(42) to pick 300 article indices.
#   4. Save the indices (not the text) — every other script re-loads text from these indices.
#
# NOTE: Indices are into the filtered article list (articles > 100 chars).
#       Run this exactly once before anything else.

import os
import json
import random

from datasets import load_dataset

# ── output path ──────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "dataset_index.json")

# ── settings ──────────────────────────────────────────────────────────────────
SEED     = 42
NUM_DOCS = 300
DATASET  = "Salesforce/wikitext"
CONFIG   = "wikitext-103-v1"
SPLIT    = "train"
MIN_CHARS = 100   # same filter as old-proj/data_utils.py


def load_all_articles():
    """Group WikiText-103 rows into full articles. Returns list of article strings."""
    print(f"Loading {DATASET} ({CONFIG}, {SPLIT})...")
    ds = load_dataset(DATASET, CONFIG, split=SPLIT)

    articles = []
    current  = []

    for row in ds:
        text = row["text"].strip()
        # A new top-level heading signals a new article
        if text.startswith("= ") and text.endswith(" =") and text.count("=") == 2:
            if current:
                articles.append(" ".join(current))
            current = [text]
        else:
            if text:
                current.append(text)

    if current:
        articles.append(" ".join(current))

    # keep only articles long enough to be useful
    articles = [a for a in articles if len(a) > MIN_CHARS]
    print(f"Found {len(articles)} usable articles after filtering.")
    return articles


def main():
    articles = load_all_articles()

    if len(articles) < NUM_DOCS:
        raise ValueError(f"Only {len(articles)} articles available, need {NUM_DOCS}.")

    # Fix seed and sample indices (NOT the text — indices only)
    random.seed(SEED)
    indices = sorted(random.sample(range(len(articles)), NUM_DOCS))

    payload = {
        "seed":     SEED,
        "dataset":  DATASET,
        "config":   CONFIG,
        "split":    SPLIT,
        "num_docs": NUM_DOCS,
        "indices":  indices,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] Saved {NUM_DOCS} shared indices to: {OUT_PATH}")
    print(f"   First 10 indices: {indices[:10]}")


if __name__ == "__main__":
    main()
