# step3_profile_llama.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Profile Llama-8B (4-bit quantized) on 300 shared docs.
#
# PREREQUISITE: step2_quant_check.py must have passed (Jaccard >= 0.95)
#               before running this script. The script checks this automatically.
#
# MODEL: unsloth/meta-llama-3.1-8B-bnb-4bit (already cached on D: drive)
# MEMORY: ~4.5GB VRAM for weights + ~1GB spike for attention matrices.
#         Ensure paging file >= 20GB if VRAM < 6GB.
#
# OUTPUTS:
#   outputs/phase1/llama-3.1-8b-4bit_patterns.pkl
#   outputs/phase1/llama-3.1-8b-4bit_patterns_summary.json

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "outputs", "phase1")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.json")
QUANT_JSON = os.path.join(OUT_DIR, "quant_check_result.json")
OFFLOAD    = os.path.join(ROOT, "offload_cache")

os.environ["HF_HOME"]             = "d:\\.cache\\huggingface"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTHONIOENCODING"]     = "utf-8"

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_ID  = "unsloth/meta-llama-3.1-8B-bnb-4bit"
MODEL_SLUG = "llama-3.1-8b-4bit"
NUM_DOCS   = 300


def check_prerequisites():
    if not os.path.exists(INDEX_PATH):
        print("[ERROR] dataset_index.json missing. Run step1_generate_index.py first.")
        sys.exit(1)
    if not os.path.exists(QUANT_JSON):
        print("[ERROR] quant_check_result.json missing. Run step2_quant_check.py first.")
        sys.exit(1)
    with open(QUANT_JSON) as f:
        result = json.load(f)
    if result.get("verdict") != "PASS":
        print(f"[ERROR] Quantization check did not pass (verdict={result.get('verdict')}).")
        print("        Llama-8B 4-bit data may be distorted. Check step2 results.")
        sys.exit(1)
    print(f"[OK] Quantization check passed (Jaccard={result['jaccard_similarity']:.3f}).")


def load_model():
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Try GPU-only first; fall back to auto (GPU+RAM) if OOM
    for device_map, tag in [({"": "cuda"}, "GPU-only"), ("auto", "auto (GPU+RAM)")]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb,
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

    print("[ERROR] Could not load Llama-8B.")
    sys.exit(1)


def main():
    check_prerequisites()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from step3_profile_shared import profile

    model, tok = load_model()
    profile(model, tok, INDEX_PATH, NUM_DOCS, OUT_DIR, MODEL_SLUG)

    del model
    torch.cuda.empty_cache()
    print(f"\n[DONE] Llama-8B profiling complete.")


if __name__ == "__main__":
    main()
