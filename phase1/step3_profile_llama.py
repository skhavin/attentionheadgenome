# step3_profile_llama.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Profile Llama-3.2-1B (native precision) on 300 shared docs.
#
# PREREQUISITE: step1_generate_index.py must have run to generate dataset_index.json.
#
# MODEL: unsloth/Llama-3.2-1B (cached on D: drive)
# MEMORY: ~2.5GB VRAM for weights. Runs comfortably in native FP16.
#
# OUTPUTS:
#   outputs/phase1/llama-3.2-1b_patterns.pkl
#   outputs/phase1/llama-3.2-1b_patterns_summary.json

import os
import sys

# Set cache directories BEFORE importing transformers
os.environ["HF_HOME"]             = "d:\\.cache\\huggingface"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTHONIOENCODING"]     = "utf-8"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "outputs", "phase1")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.json")

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_ID   = "unsloth/Llama-3.2-1B"
MODEL_SLUG = "llama-3.2-1b"
NUM_DOCS   = 300


def check_prerequisites():
    if not os.path.exists(INDEX_PATH):
        print("[ERROR] dataset_index.json missing. Run step1_generate_index.py first.")
        sys.exit(1)
    print("[OK] Prerequisites satisfied.")


def load_model():
    print(f"Loading {MODEL_ID} in native precision...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  Using precision: {dtype}")

    # Try GPU-only first; fall back to auto (GPU+RAM) if OOM
    for device_map, tag in [({"": "cuda"}, "GPU-only"), ("auto", "auto (GPU+RAM)")]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            model.eval()
            print(f"  Loaded [{tag}]")
            return model, tok
        except Exception as e:
            print(f"  Failed [{tag}]: {str(e)[:120]}")
            torch.cuda.empty_cache()

    print("[ERROR] Could not load Llama-3.2-1B.")
    sys.exit(1)


def main():
    check_prerequisites()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from step3_profile_shared import profile

    model, tok = load_model()
    profile(model, tok, INDEX_PATH, NUM_DOCS, OUT_DIR, MODEL_SLUG)

    del model
    torch.cuda.empty_cache()
    print(f"\n[DONE] Llama-3.2-1B profiling complete.")


if __name__ == "__main__":
    main()
