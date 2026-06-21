# step3_profile_qwen.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Profile Qwen-0.5B and Qwen-1.5B on 300 shared docs.
#          Runs each model sequentially to avoid VRAM conflicts.
#
# MODELS:
#   Qwen/Qwen2.5-0.5B  — ~1GB FP16, fast
#   Qwen/Qwen2.5-1.5B  — ~3GB FP16; if OOM, falls back to 4-bit
#
# OUTPUTS (per model):
#   outputs/phase1/{slug}_patterns.pkl
#   outputs/phase1/{slug}_patterns_summary.json

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(ROOT, "outputs", "phase1")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.json")

os.environ["HF_HOME"]             = "d:\\.cache\\huggingface"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

# ── settings ──────────────────────────────────────────────────────────────────
NUM_DOCS = 300

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "qwen2.5-0.5b"),
    ("Qwen/Qwen2.5-1.5B", "qwen2.5-1.5b"),
]


def load_model_fp16(model_id):
    """Load model in FP16. Returns (model, tokenizer) or raises."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  Loading {model_id} in {'BF16' if dtype == torch.bfloat16 else 'FP16'}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tok


def load_model_4bit(model_id):
    """Fallback: load in 4-bit if FP16 OOMs."""
    print(f"  FP16 OOM — retrying {model_id} in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tok


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] dataset_index.json missing. Run step1_generate_index.py first.")
        sys.exit(1)

    # Import shared profiler
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from step3_profile_shared import profile

    for model_id, slug in MODELS:
        print(f"\n{'='*50}")
        print(f"MODEL: {model_id}  slug={slug}")
        print(f"{'='*50}")

        # Try FP16 first, fall back to 4-bit on OOM
        try:
            model, tok = load_model_fp16(model_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model, tok = load_model_4bit(model_id)

        profile(model, tok, INDEX_PATH, NUM_DOCS, OUT_DIR, slug)

        # Free VRAM before loading the next model
        del model
        torch.cuda.empty_cache()
        print(f"  [DONE] {slug}")


if __name__ == "__main__":
    main()
