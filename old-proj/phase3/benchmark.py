# Benchmark proactive KV eviction vs full attention.
# Protocol: prefill context -> prune KV cache -> evaluate perplexity on held-out tokens.
# The pruned cache means the model has LESS context, so predictions get worse = higher PPL.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from phase2.predict_prototypes import predict_prototypes
from phase2.build_retention_mask import predict_retention_mask
from phase3.kv_cache_wrapper import apply_retention_mask
from config import (MODEL_NAME, DEVICE, USE_FP16, PHASE3_DIR, PROTOTYPES_PATH,
                    MAX_SEQ_LEN, KV_BUDGETS, NUM_BENCHMARK_DOCS)

# How many tokens to hold out for perplexity evaluation
EVAL_TOKENS = 50

def prune_kv_to_budget(past_kv, budget, device):
    """Simple uniform pruning: keep first 4 (sinks) + evenly spaced + last tokens."""
    seq_len = past_kv[0][0].shape[2]
    if seq_len <= budget:
        return past_kv  # nothing to prune

    # Always keep: first 4 (attention sinks) + last 4 (recency)
    sink_count = min(4, budget // 4)
    recent_count = min(4, budget // 4)
    middle_budget = budget - sink_count - recent_count

    sinks = list(range(sink_count))
    recents = list(range(seq_len - recent_count, seq_len))

    # Evenly space the rest
    middle_range = list(range(sink_count, seq_len - recent_count))
    if middle_budget > 0 and len(middle_range) > 0:
        step = max(1, len(middle_range) // middle_budget)
        middles = middle_range[::step][:middle_budget]
    else:
        middles = []

    indices = sorted(set(sinks + middles + recents))[:budget]
    indices_t = torch.tensor(indices, dtype=torch.long, device=device)

    pruned = []
    for layer_kv in past_kv:
        k, v = layer_kv
        pruned.append((k.index_select(2, indices_t), v.index_select(2, indices_t)))
    return tuple(pruned)


def eval_ppl_with_kv(model, input_ids, past_kv, eval_start):
    """Autoregressively evaluate perplexity on tokens from eval_start onward, using past_kv."""
    targets = input_ids[:, eval_start:]
    gen_len = targets.shape[1]
    if gen_len < 5:
        return None

    nlls = []
    # Feed the token just before eval_start to start
    next_token = input_ids[:, eval_start - 1:eval_start]

    for i in range(gen_len):
        output = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = output.past_key_values
        logits = output.logits[:, -1, :]
        target = targets[:, i]
        nll = torch.nn.functional.cross_entropy(logits, target).item()
        nlls.append(nll)
        next_token = target.unsqueeze(0)

    return np.exp(np.mean(nlls))


def measure_full_attention(model, tokenizer, text):
    """Full attention baseline — no pruning."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 2)
    if eval_start < 10:
        return None

    with torch.no_grad():
        # Prefill context (all tokens up to eval_start - 1)
        context = input_ids[:, :eval_start - 1]
        output = model(context, use_cache=True)
        return eval_ppl_with_kv(model, input_ids, output.past_key_values, eval_start)


def measure_proactive(model, tokenizer, text, prototypes, budget):
    """Proactive KV eviction — prune cache using predicted prototypes."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 2)
    if eval_start < 10:
        return None

    with torch.no_grad():
        # Prefill full context
        context = input_ids[:, :eval_start - 1]
        output = model(context, use_cache=True)
        past_kv = output.past_key_values

        # Proactive pruning: predict which tokens matter per head, then prune
        predictions = predict_prototypes(None, prototypes)
        masks = predict_retention_mask(prototypes, predictions, context.shape[1], budget)
        pruned_kv = apply_retention_mask(past_kv, masks, budget, device=DEVICE)

        return eval_ppl_with_kv(model, input_ids, pruned_kv, eval_start)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Proactive KV-eviction short-doc benchmark")
    parser.add_argument("--budgets", type=int, nargs="+", default=KV_BUDGETS,
                        help="KV cache budgets to evaluate")
    args = parser.parse_args()
    budgets = args.budgets

    os.makedirs(PHASE3_DIR, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)

    articles = load_articles(split="validation", max_articles=NUM_BENCHMARK_DOCS)

    save_path = os.path.join(PHASE3_DIR, "benchmark_results.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results from {save_path}")
    else:
        results = {}

    # Full attention baseline (skip if already done or not evaluating all budgets)
    if "full" not in results:
        print("\n--- Full Attention (no pruning) ---")
        ppls = []
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for text in tqdm(articles, desc="Full attn"):
            ppl = measure_full_attention(model, tokenizer, text)
            if ppl:
                ppls.append(ppl)
        elapsed = time.time() - start
        vram = torch.cuda.max_memory_allocated() / 1e6
        results["full"] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
        print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")
    else:
        print(f"  [SKIP] full attention already evaluated (PPL={results['full']['ppl']:.2f})")

    # Proactive eviction at each budget
    for budget in budgets:
        print(f"\n--- Proactive (budget={budget}) ---")
        ppls = []
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for text in tqdm(articles, desc=f"Proactive B={budget}"):
            ppl = measure_proactive(model, tokenizer, text, prototypes, budget)
            if ppl:
                ppls.append(ppl)
        elapsed = time.time() - start
        vram = torch.cuda.max_memory_allocated() / 1e6
        results[f"proactive_{budget}"] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
        print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {save_path}")

if __name__ == "__main__":
    main()
