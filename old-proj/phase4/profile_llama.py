# Profile large RoPE models (Qwen2.5-7B / LLaMA 3.1 8B) in 4-bit — attention head specialization at scale.
# Qwen2.5-7B is tried first (already cached). LLaMA is fallback.
# Shows that prototype clusters form in large RoPE-based models just like in GPT-2.

import sys, os
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["SAFETENSORS_FAST_GPU"] = "1"  # Load directly to GPU, bypasses Windows paging file mmap
os.environ["PYTHONIOENCODING"] = "utf-8"   # Prevent silent crash on non-ASCII error messages
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

# unsloth LLaMA 3.1 8B 4-bit is fully downloaded (5.31GB blob confirmed)
MODELS_TO_TRY = [
    "unsloth/meta-llama-3.1-8B-bnb-4bit",
    "meta-llama/Meta-Llama-3.1-8B",
]
PROFILE_DOCS = 50       # fewer docs since model is bigger
PROFILE_SEQ_LEN = 512   # 512 is minimum for meaningful attention pattern diversity
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")

# 4-bit quantization config
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def extract_patterns(model, tokenizer, text):
    """Extract attention offset patterns from one document.
    Requires adequate Windows paging file (20-40GB on SSD) to handle
    the ~1GB attention matrix spike alongside 4.5GB model weights.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=PROFILE_SEQ_LEN)
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
            counts = np.bincount(abs_off.astype(int), minlength=PROFILE_SEQ_LEN)[:PROFILE_SEQ_LEN]
            total = counts.sum()
            patterns[(layer_idx, head_idx)] = counts / total if total > 0 else counts.astype(float)

    return patterns


def try_load_model(model_name):
    """Load model. Tries GPU-only first (avoids paging file OOM), then disk offload."""
    strategies = [
        ({"device_map": {"":"cuda"}},               "GPU-only"),
        ({"device_map": "auto"},                     "auto (GPU+RAM)"),
        ({"device_map": "auto",
          "offload_folder": OFFLOAD_DIR,
          "offload_state_dict": True},               "disk offload"),
    ]
    for extra_kwargs, tag in strategies:
        try:
            print(f"Loading {model_name} in 4-bit [{tag}]...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            os.makedirs(OFFLOAD_DIR, exist_ok=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=BNB_CONFIG,
                trust_remote_code=True,
                attn_implementation="eager",
                **extra_kwargs,
            )
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Loaded successfully [{tag}]")
            return model, tokenizer, model_name
        except Exception as e:
            err = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"  Failed [{tag}]: {err}")
    return None


def check_swap():
    """Verify paging file is large enough before attempting LLaMA 8B profiling."""
    import psutil
    swap = psutil.swap_memory()
    swap_gb = swap.total / 1e9
    print(f"Swap/paging file available: {swap_gb:.1f} GB")
    if swap_gb < 18.0:
        print("ERROR: Paging file too small. Set D: to Initial=20000MB, Max=40000MB and RESTART.")
        print("       LLaMA 8B 4-bit weights (4.5GB) + output_attentions spike (~1GB) = 5.5GB")
        print("       You have 4GB VRAM, so ~1.5GB must page to RAM. Need 18GB+ paging file.")
        return False
    print("  Paging file OK.")
    return True


def main():
    if not check_swap():
        return
    os.makedirs(PHASE4_DIR, exist_ok=True)

    # Try loading from the models to try list
    result = None
    for model_name in MODELS_TO_TRY:
        result = try_load_model(model_name)
        if result is not None:
            break

    if result is None:
        print("ERROR: Could not load any 7B+ model. Check HuggingFace access tokens.")
        return

    model, tokenizer, model_name = result
    short_name = model_name.split("/")[-1]
    print(f"Using: {model_name}")

    articles = load_articles(split="train", max_articles=PROFILE_DOCS)

    all_patterns = []
    for text in tqdm(articles, desc=f"Profiling {short_name}"):
        p = extract_patterns(model, tokenizer, text)
        if p is not None:
            all_patterns.append(p)

    # Save patterns
    save_path = os.path.join(PHASE4_DIR, f"{short_name.lower()}_attention_patterns.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_patterns, f)
    print(f"Saved {len(all_patterns)} patterns to {save_path}")

    # Cluster and report
    keys = sorted(all_patterns[0].keys())
    print(f"\nModel: {model_name}")
    print(f"Layers: {max(l for l,h in keys)+1}, Heads per layer: {max(h for l,h in keys)+1}")
    print(f"\nClustering results (first 10 heads, {len(all_patterns)} docs):")

    cluster_inertias = []
    for layer, head in keys[:10]:
        data = np.array([d[(layer, head)] for d in all_patterns if (layer, head) in d])
        k = min(NUM_CLUSTERS, len(data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        cluster_inertias.append(kmeans.inertia_)
        print(f"  Layer {layer:2d}, Head {head:2d}: inertia={kmeans.inertia_:.4f} <-- {'CLUSTERS EXIST' if kmeans.inertia_ < 1.0 else 'weak clusters'}")

    # Locality distribution
    all_locality = []
    for (layer, head) in keys:
        data = np.array([d[(layer, head)] for d in all_patterns if (layer, head) in d])
        mean_pattern = data.mean(axis=0)
        all_locality.append(mean_pattern[:10].sum())

    print(f"\n{short_name} locality: mean={np.mean(all_locality):.3f}, std={np.std(all_locality):.3f}")
    print(f"Avg cluster inertia (first 10 heads): {np.mean(cluster_inertias):.4f}")

    # Save summary
    summary = {
        "model_name": model_name,
        "short_name": short_name,
        "num_docs": len(all_patterns),
        "num_layers": max(l for l, h in keys) + 1,
        "num_heads": max(h for l, h in keys) + 1,
        "locality_mean": float(np.mean(all_locality)),
        "locality_std": float(np.std(all_locality)),
        "avg_inertia": float(np.mean(cluster_inertias)),
        "all_locality": all_locality,
    }
    summary_path = os.path.join(PHASE4_DIR, f"{short_name.lower()}_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
