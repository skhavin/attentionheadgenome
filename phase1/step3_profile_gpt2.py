# step3_profile_gpt2.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Profile GPT-2 Medium on 300 shared docs from dataset_index.json.
#
# WHY RE-RUN: The old-proj pkl used sequential article[0:500]. Our shared index
#             samples from the full 29K article pool, so very few indices overlap.
#             Re-profiling is the only way to get sample-symmetric data.
#
# MEMORY: GPT-2 Medium is 345M FP16 — fits comfortably in 4GB VRAM.
#
# OUTPUTS:
#   outputs/phase1/gpt2-medium_patterns.pkl
#   outputs/phase1/gpt2-medium_patterns_summary.json

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "outputs", "phase1")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.json")

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_ID   = "gpt2-medium"
MODEL_SLUG = "gpt2-medium"
NUM_DOCS   = 300


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] dataset_index.json missing. Run step1_generate_index.py first.")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from step3_profile_shared import profile

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
    )
    # GPT-2 is small enough to run in FP16 on GPU or full precision on CPU
    if torch.cuda.is_available():
        model = model.half().cuda()
    model.eval()

    profile(model, tok, INDEX_PATH, NUM_DOCS, OUT_DIR, MODEL_SLUG)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n[DONE] GPT-2 Medium profiling complete.")


if __name__ == "__main__":
    main()
