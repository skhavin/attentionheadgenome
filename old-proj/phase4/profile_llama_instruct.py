# Profile LLaMA 3.1 8B Instruct — generate prototypes for kvpress leaderboard evaluation.
#
# Uses the same BnB 4-bit config and attention pattern extraction pipeline as
# phase4/profile_llama.py, but targets the Instruct variant.
#
# Output:
#   outputs/phase4/meta-llama-3.1-8b-instruct-bnb-4bit_prototypes.pkl
#
# Usage:
#   python phase4/profile_llama_instruct.py

import sys, os
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.cluster import KMeans
from data_utils import load_articles
from config import DEVICE, PHASE4_DIR, TOP_K_ATTENTION, NUM_CLUSTERS

# Instruct variant — unsloth 4-bit is pre-quantized (no VRAM spike from conversion)
MODELS_TO_TRY = [
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  # preferred: already 4-bit
    "meta-llama/Meta-Llama-3.1-8B-Instruct",         # fallback: quantize on-the-fly
]
PROFILE_DOCS   = 50
PROFILE_SEQ_LEN = 512
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def try_load_model(model_name):
    strategies = [
        ({"device_map": {"": "cuda"}},  "GPU-only"),
        ({"device_map": "auto"},         "auto (GPU+RAM)"),
        ({"device_map": "auto",
          "offload_folder": OFFLOAD_DIR,
          "offload_state_dict": True},   "disk offload"),
    ]
    for extra_kwargs, tag in strategies:
        try:
            print(f"  Loading {model_name} [{tag}]...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            os.makedirs(OFFLOAD_DIR, exist_ok=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=BNB_CONFIG,
                trust_remote_code=True,
                attn_implementation="eager",   # needed for output_attentions=True
                **extra_kwargs,
            )
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Loaded [{tag}]")
            return model, tokenizer, model_name
        except Exception as e:
            err = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"  Failed [{tag}]: {err}")
    return None


def extract_patterns(model, tokenizer, text):
    """Extract per-(layer, head) attention offset histograms from one document."""
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=PROFILE_SEQ_LEN
    )
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    seq_len = tokens["input_ids"].shape[1]
    if seq_len < 10:
        return None

    with torch.no_grad():
        output = model(**tokens, output_attentions=True)

    patterns = {}
    q_indices = torch.arange(seq_len).view(1, seq_len, 1)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)

    for layer_idx, layer_attn in enumerate(output.attentions):
        if layer_attn is None:
            continue
        attn = layer_attn[0].float().cpu()
        attn = attn.masked_fill(mask, -float("inf"))

        k = min(TOP_K_ATTENTION, seq_len)
        _, top_indices = attn.topk(k, dim=2)
        rel_offsets = top_indices - q_indices

        for head_idx in range(attn.shape[0]):
            offsets = rel_offsets[head_idx].flatten().numpy()
            offsets = offsets[offsets <= 0]
            abs_off = np.abs(offsets)
            abs_off = np.minimum(abs_off, PROFILE_SEQ_LEN - 1)
            counts = np.bincount(
                abs_off.astype(int), minlength=PROFILE_SEQ_LEN
            )[:PROFILE_SEQ_LEN]
            total = counts.sum()
            patterns[(layer_idx, head_idx)] = (
                counts / total if total > 0 else counts.astype(float)
            )

    return patterns


def build_prototypes(all_patterns, num_clusters):
    """K-Means cluster patterns per (layer, head) → prototype centroids."""
    keys = sorted(all_patterns[0].keys())
    prototypes = {}
    print(f"\nClustering {len(keys)} (layer, head) pairs into {num_clusters} prototypes each...")
    for layer, head in tqdm(keys, desc="Clustering"):
        data = np.array(
            [d[(layer, head)] for d in all_patterns if (layer, head) in d]
        )
        if len(data) < num_clusters:
            continue
        k = min(num_clusters, len(data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        prototypes[(layer, head)] = {
            "centroids": kmeans.cluster_centers_,   # shape: (k, seq_len)
            "labels":    kmeans.labels_,
            "inertia":   kmeans.inertia_,
        }
    return prototypes


def main():
    os.makedirs(PHASE4_DIR, exist_ok=True)

    # Load model
    result = None
    for mn in MODELS_TO_TRY:
        result = try_load_model(mn)
        if result is not None:
            break

    if result is None:
        print("ERROR: Could not load any Instruct model. "
              "Run: huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
        return

    model, tokenizer, model_name = result
    short_name = model_name.split("/")[-1].lower()
    print(f"\nUsing: {model_name}  ({short_name})")

    # Check output paths
    patterns_path  = os.path.join(PHASE4_DIR, f"{short_name}_attention_patterns.pkl")
    prototypes_path = os.path.join(PHASE4_DIR, f"{short_name}_prototypes.pkl")

    if os.path.exists(prototypes_path):
        print(f"\nPrototypes already exist at: {prototypes_path}")
        print("Delete the file and re-run to regenerate. Exiting.")
        return

    # Load calibration corpus (WikiText articles, same as base model profiling)
    articles = load_articles(split="train", max_articles=PROFILE_DOCS)
    print(f"\nProfiling on {len(articles)} WikiText articles (seq_len={PROFILE_SEQ_LEN})...")

    all_patterns = []
    for text in tqdm(articles, desc=f"Extracting patterns [{short_name}]"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        p = extract_patterns(model, tokenizer, text)
        if p is not None:
            all_patterns.append(p)

    print(f"Extracted patterns from {len(all_patterns)} documents.")

    # Save raw patterns
    with open(patterns_path, "wb") as f:
        pickle.dump(all_patterns, f)
    print(f"Saved raw patterns: {patterns_path}")

    # Build and save prototypes
    prototypes = build_prototypes(all_patterns, NUM_CLUSTERS)
    with open(prototypes_path, "wb") as f:
        pickle.dump(prototypes, f)
    print(f"\nSaved prototypes ({len(prototypes)} heads): {prototypes_path}")
    print("\nDone. Use this path in ProactiveCachePress:")
    print(f'  prototype_path="{prototypes_path}"')


if __name__ == "__main__":
    main()
