# LLaMA 3.1 8B eviction benchmark — RoPE models don't have the absolute position problem.
# Tests proactive vs StreamingLLM at budgets 128/256/512/1024.
# Skips H2O because output_attentions on 7B is too expensive on 4GB VRAM.
#
# Usage:
#   python phase5/benchmark_llama.py --dataset wikitext
#   python phase5/benchmark_llama.py --dataset pg19 --budgets 128 256 512
#
# Features:
#   - Saves incrementally after every (method, budget) combo
#   - Resumes automatically from the last checkpoint if interrupted
#   - Results go to outputs/phase5/llama_<dataset>_benchmark.pkl

import sys, os, argparse
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.cluster import KMeans
from data_utils import load_concatenated_articles
from config import PHASE4_DIR, NUM_CLUSTERS

# unsloth LLaMA 3.1 8B 4-bit is fully downloaded (5.31GB blob confirmed)
MODELS_TO_TRY = [
    "unsloth/meta-llama-3.1-8B-bnb-4bit",
    "meta-llama/Meta-Llama-3.1-8B",
]
BENCH_SEQ_LEN  = 1024   # tokens per chunk
EVAL_TOKENS    = 100    # tokens used for PPL eval per chunk
NUM_BENCH_DOCS = 10     # WikiText: number of concatenated-article docs
PROFILE_SEQ_LEN = 512   # used when building prototypes from patterns
NUM_BOOKS      = 2      # PG-19: number of books (reduced for local hardware speed)
CHUNKS_PER_BOOK = 3      # PG-19: chunks per book (reduced for local hardware speed)

OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase5")

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
# -----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA KV-eviction benchmark")
    parser.add_argument(
        "--dataset", choices=["wikitext", "pg19"], default="wikitext",
        help="Dataset to benchmark on (default: wikitext)"
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[128, 256, 512, 1024],
        help="KV cache budgets to test (default: 128 256 512 1024)"
    )
    return parser.parse_args()


# ----- Prototype helpers -----------------------------------------------

def build_prototypes_from_patterns(all_patterns, num_clusters=4, max_seq_len=512):
    keys = sorted(all_patterns[0].keys())
    prototypes = {}
    for (layer, head) in keys:
        data = np.array([d[(layer, head)] for d in all_patterns if (layer, head) in d])
        k = min(num_clusters, len(data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        prototypes[(layer, head)] = {"centroids": kmeans.cluster_centers_, "labels": kmeans.labels_}
    return prototypes


def predict_and_mask(prototypes, seq_len, budget):
    sink_count = min(4, budget)
    # Proportional recency: allocate 50% of the budget to local context
    recent_count = min(seq_len - sink_count, budget // 2)
    semantic_budget = max(0, budget - sink_count - recent_count)

    # 1. Select sinks and recency indices
    sinks = set(range(sink_count))
    recents = set(range(seq_len - recent_count, seq_len))

    # 2. Score intermediate tokens using profiled prototypes
    scores = np.zeros(seq_len, dtype=np.float64)
    if prototypes is not None:
        for (layer, head) in prototypes.keys():
            centroid = prototypes[(layer, head)]["centroids"][0]
            max_d = min(len(centroid), seq_len)
            if max_d > 0:
                cumsum = np.cumsum(centroid[:max_d])
                for p in range(seq_len):
                    reach = min(max_d, seq_len - p)
                    if reach > 0:
                        scores[p] += cumsum[reach - 1]

    # Zero out already selected sink/recent indices to avoid re-selection
    for idx in list(sinks) + list(recents):
        if idx < seq_len:
            scores[idx] = -np.inf

    # Pick top semantic tokens
    allowed_indices = [i for i in range(seq_len) if scores[i] != -np.inf]
    if len(allowed_indices) > 0 and semantic_budget > 0:
        top_semantic = np.argsort(scores[allowed_indices])[-semantic_budget:]
        semantic_indices = [allowed_indices[i] for i in top_semantic]
    else:
        semantic_indices = []

    final_idx = sorted(list(sinks.union(recents).union(set(semantic_indices))))
    return final_idx


def streaming_llm_indices(seq_len, budget):
    sink_count = min(4, budget)
    recent_count = budget - sink_count
    sinks = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    return sorted(set(sinks + recents))[:budget]


def _to_tuple_kv(past_kv):
    """Normalize DynamicCache or legacy tuple to tuple of (k,v)."""
    if hasattr(past_kv, "to_legacy_cache"):
        return past_kv.to_legacy_cache()
    return tuple(past_kv)


def _to_dynamic_cache(kv_tuple):
    """Convert (k,v) tuple back to DynamicCache for models that require it."""
    try:
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(kv_tuple)
    except ImportError:
        return kv_tuple  # Old transformers — tuple is fine


def prune_kv(past_kv, indices, device):
    indices_t = torch.tensor(indices, dtype=torch.long, device=device)
    pruned = tuple(
        (k.index_select(2, indices_t), v.index_select(2, indices_t))
        for k, v in _to_tuple_kv(past_kv)
    )
    return _to_dynamic_cache(pruned)


def eval_ppl(model, input_ids, past_kv, eval_start):
    targets = input_ids[:, eval_start:]
    gen_len = targets.shape[1]
    if gen_len < 5:
        return None
    nlls = []
    next_token = input_ids[:, eval_start - 1:eval_start]
    for i in range(gen_len):
        output = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = output.past_key_values
        logits = output.logits[:, -1, :]
        nll = torch.nn.functional.cross_entropy(logits, targets[:, i]).item()
        nlls.append(nll)
        next_token = targets[:, i].unsqueeze(0)
    return np.exp(np.mean(nlls))


# ----- Model loading ---------------------------------------------------

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
                **extra_kwargs,
            )
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Loaded successfully [{tag}]")
            return model, tokenizer, model_name
        except Exception as e:
            print(f"  Failed [{tag}]: {e}")
    return None


# ----- Dataset loaders -------------------------------------------------

def load_wikitext_chunks(tokenizer):
    print("Loading WikiText-103 validation docs...")
    docs = load_concatenated_articles(split="validation", articles_per_doc=10, max_docs=NUM_BENCH_DOCS)
    chunks = []
    for text in docs:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=BENCH_SEQ_LEN)["input_ids"]
        chunks.append(ids)
    print(f"  {len(chunks)} chunks loaded.")
    return chunks


def load_pg19_chunks(tokenizer):
    from datasets import load_dataset
    print("Loading PG-19 test books (streaming)...")
    ds = load_dataset("emozilla/pg19", split="test", streaming=True).take(NUM_BOOKS)
    chunks = []
    for i, example in enumerate(ds):
        text = example["text"]
        all_tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        total = len(all_tokens)
        if total < BENCH_SEQ_LEN:
            continue
        step = max(BENCH_SEQ_LEN, (total - BENCH_SEQ_LEN) // CHUNKS_PER_BOOK)
        book_chunks = []
        for start in range(0, total - BENCH_SEQ_LEN, step):
            if len(book_chunks) >= CHUNKS_PER_BOOK:
                break
            book_chunks.append(all_tokens[start:start + BENCH_SEQ_LEN].unsqueeze(0))
        chunks.extend(book_chunks)
        print(f"  Book {i}: {len(text):,} chars -> {len(book_chunks)} chunks")
    print(f"  Total: {len(chunks)} chunks")
    return chunks


# ----- Incremental save / load checkpoint ------------------------------

def load_checkpoint(save_path):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming from checkpoint: {save_path}")
        print(f"  Already done: {list(results.keys())}")
        return results
    return {}


def save_checkpoint(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


# ----- Evaluation loop -------------------------------------------------

def run_eval(model, chunks, device, method, budget, prototypes):
    """Run one (method, budget) combo over all chunks, return metrics dict."""
    ppls = []
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    label = f"{method.capitalize()} B={budget}" if method != "full" else "Full"
    for chunk in tqdm(chunks, desc=label):
        input_ids = chunk.to(device)
        seq_len = input_ids.shape[1]
        eval_start = seq_len - min(EVAL_TOKENS, seq_len // 4)
        if eval_start < 20:
            continue
        with torch.no_grad():
            context = input_ids[:, :eval_start - 1]
            output = model(context, use_cache=True)
            past_kv = output.past_key_values
            if method != "full":
                ctx_len = context.shape[1]
                if method == "proactive":
                    indices = predict_and_mask(prototypes, ctx_len, min(budget, ctx_len))
                else:  # streamingllm
                    indices = streaming_llm_indices(ctx_len, min(budget, ctx_len))
                past_kv = prune_kv(past_kv, indices, device)
            ppl = eval_ppl(model, input_ids, past_kv, eval_start)
            if ppl:
                ppls.append(ppl)
    elapsed = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1e6
    return {"ppl": float(np.mean(ppls)), "vram_mb": vram, "time_s": elapsed}


# ----- Main ------------------------------------------------------------

def main():
    args = parse_args()
    dataset   = args.dataset
    budgets   = args.budgets

    save_path = os.path.join(OUTPUT_DIR, f"llama_{dataset}_benchmark.pkl")
    print(f"\n{'='*60}")
    print(f"  LLaMA Benchmark | dataset={dataset} | budgets={budgets}")
    print(f"  Save path: {save_path}")
    print(f"{'='*60}\n")

    # ---- Load checkpoint (resume support) ----
    results = load_checkpoint(save_path)

    # ---- Load model ----
    result = None
    for model_name in MODELS_TO_TRY:
        result = try_load_model(model_name)
        if result is not None:
            break
    if result is None:
        print("ERROR: Could not load any LLaMA model.")
        return
    model, tokenizer, model_name = result
    short_name = model_name.split("/")[-1]
    device = next(model.parameters()).device

    # ---- Load or build prototypes ----
    proto_path   = os.path.join(PHASE4_DIR, f"{short_name.lower()}_prototypes.pkl")
    pattern_path = os.path.join(PHASE4_DIR, f"{short_name.lower()}_attention_patterns.pkl")

    if os.path.exists(proto_path):
        with open(proto_path, "rb") as f:
            prototypes = pickle.load(f)
        print(f"Loaded prototypes from {proto_path}")
    elif os.path.exists(pattern_path):
        with open(pattern_path, "rb") as f:
            patterns = pickle.load(f)
        prototypes = build_prototypes_from_patterns(patterns, NUM_CLUSTERS, PROFILE_SEQ_LEN)
        with open(proto_path, "wb") as f:
            pickle.dump(prototypes, f)
        print(f"Built prototypes from patterns -> {proto_path}")
    else:
        print(f"ERROR: No prototypes or patterns found at {proto_path}")
        print(f"       Run: python phase4/profile_llama.py")
        return

    # ---- Load dataset ----
    if dataset == "wikitext":
        chunks = load_wikitext_chunks(tokenizer)
    else:
        chunks = load_pg19_chunks(tokenizer)

    # ---- Full attention (skip if already done) ----
    if "full" not in results:
        print(f"\n--- Full Attention ({short_name}, {dataset.upper()}) ---")
        results["full"] = run_eval(model, chunks, device, "full", None, None)
        save_checkpoint(results, save_path)
        r = results["full"]
        print(f"  PPL={r['ppl']:.2f}, VRAM={r['vram_mb']:.0f}MB, Time={r['time_s']:.1f}s")
    else:
        r = results["full"]
        print(f"  [SKIP] full: PPL={r['ppl']:.2f} (already done)")

    # ---- Method x budget grid ----
    METHODS = ["proactive", "streamingllm"]
    for budget in budgets:
        for method in METHODS:
            key = f"{method}_{budget}"
            if key in results:
                r = results[key]
                print(f"  [SKIP] {key}: PPL={r['ppl']:.2f} (already done)")
                continue
            print(f"\n--- {method.capitalize()} (budget={budget}, {short_name}, {dataset.upper()}) ---")
            results[key] = run_eval(model, chunks, device, method, budget, prototypes)
            save_checkpoint(results, save_path)   # save after EVERY combo
            r = results[key]
            print(f"  PPL={r['ppl']:.2f}, VRAM={r['vram_mb']:.0f}MB, Time={r['time_s']:.1f}s")

    # ---- Print final table ----
    print(f"\n{'='*56}")
    print(f"  {short_name} | {dataset.upper()} | Final Results")
    print(f"{'='*56}")
    print(f"{'Method':<20} {'Budget':>6} {'PPL':>8} {'VRAM(MB)':>9} {'Time(s)':>8}")
    print("-" * 56)
    r = results["full"]
    print(f"{'Full Attention':<20} {'all':>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")
    for budget in budgets:
        print("-" * 56)
        for method, label in [("streamingllm", "StreamingLLM"), ("proactive", "Proactive (ours)")]:
            key = f"{method}_{budget}"
            if key in results:
                r = results[key]
                print(f"{label:<20} {budget:>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
