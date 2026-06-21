# step3_profile_gpt2.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Profile GPT-2 Medium on 300 shared docs.
#          GPT-2 already has 500-doc data in old-proj/outputs/phase1/attention_patterns.pkl.
#          This script TRUNCATES that existing data to the first 300 shared indices
#          instead of re-running the model — saving ~10 minutes.
#
# OUTPUTS:
#   outputs/phase1/gpt2-medium_patterns.pkl
#   outputs/phase1/gpt2-medium_patterns_summary.json

import os
import sys
import json
import pickle
import numpy as np
from datasets import load_dataset

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(ROOT, "outputs", "phase1")
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH    = os.path.join(OUT_DIR, "dataset_index.json")
OLD_PKL       = os.path.join(ROOT, "old-proj", "outputs", "phase1", "attention_patterns.pkl")
OUT_PKL       = os.path.join(OUT_DIR, "gpt2-medium_patterns.pkl")
OUT_JSON      = os.path.join(OUT_DIR, "gpt2-medium_patterns_summary.json")

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_SLUG = "gpt2-medium"
NUM_DOCS   = 300
SEQ_LEN    = 512   # same as used during original profiling


def load_all_articles_for_indexing():
    """Rebuild the same article list that was used when the old pkl was created."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    articles = []
    current  = []
    for row in ds:
        text = row["text"].strip()
        if text.startswith("= ") and text.endswith(" =") and text.count("=") == 2:
            if current:
                articles.append(" ".join(current))
            current = [text]
        elif text:
            current.append(text)
    if current:
        articles.append(" ".join(current))
    return [a for a in articles if len(a) > 100]


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] dataset_index.json not found. Run step1_generate_index.py first.")
        sys.exit(1)
    if not os.path.exists(OLD_PKL):
        print(f"[ERROR] old-proj pkl not found at: {OLD_PKL}")
        sys.exit(1)

    with open(INDEX_PATH) as f:
        cfg = json.load(f)
    shared_indices = cfg["indices"][:NUM_DOCS]   # the 300 shared article positions

    # The old pkl stored patterns in order of the first 500 articles loaded by
    # old-proj/data_utils.py (sequential, not randomly sampled).
    # We need to find which old pkl entries correspond to our shared indices.
    #
    # Strategy: rebuild the article list in the same order data_utils.py would,
    # then identify which old-pkl position each shared index corresponds to.
    #
    # Old profiling used articles[:500] sequentially. If a shared index < 500,
    # it maps directly to old_patterns[shared_index]. Otherwise we skip it.

    print("Loading old GPT-2 patterns pkl...")
    with open(OLD_PKL, "rb") as f:
        old_patterns = pickle.load(f)   # list of 500 dicts

    print(f"  Old pkl has {len(old_patterns)} docs.")

    # Select the subset of old_patterns that correspond to shared_indices.
    # old_patterns[i] = patterns for the i-th article in data_utils sequential order.
    selected = []
    skipped  = 0
    for idx in shared_indices:
        if idx < len(old_patterns):
            selected.append(old_patterns[idx])
        else:
            skipped += 1

    print(f"  Selected {len(selected)} docs from old pkl (skipped {skipped} out-of-range).")

    if len(selected) < 10:
        print("[WARN] Very few docs matched. Consider re-running the full GPT-2 profiler.")

    # ── save pkl ──────────────────────────────────────────────────────────
    with open(OUT_PKL, "wb") as f:
        pickle.dump(selected, f)
    print(f"  Saved pkl -> {OUT_PKL}")

    # ── save json summary (mean per head) ─────────────────────────────────
    if selected:
        keys    = sorted(selected[0].keys())
        summary = {}
        for layer, head in keys:
            hists = [d[(layer, head)] for d in selected if (layer, head) in d]
            mean  = np.mean(hists, axis=0).tolist()
            summary[f"{layer}_{head}"] = mean

        with open(OUT_JSON, "w") as f:
            json.dump({
                "model_slug": MODEL_SLUG,
                "num_docs":   len(selected),
                "seq_len":    SEQ_LEN,
                "source":     "truncated from old-proj 500-doc pkl",
                "heads":      summary,
            }, f)
        print(f"  Saved json -> {OUT_JSON}")

    print(f"\n[DONE] GPT-2 profiling complete. {len(selected)} docs used.")


if __name__ == "__main__":
    main()
