# Run StreamingLLM and H2O baselines at the same budgets.
# Same evaluation protocol as benchmark.py: prefill -> prune KV -> evaluate held-out tokens.
# StreamingLLM: keep first 4 (sinks) + last (budget-4) tokens in KV cache.
# H2O: run full attention, keep heavy-hitter tokens (highest cumulative attention received).

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from config import (MODEL_NAME, DEVICE, USE_FP16, PHASE3_DIR,
                    MAX_SEQ_LEN, KV_BUDGETS, NUM_BENCHMARK_DOCS)

EVAL_TOKENS = 50


def prune_kv_by_indices(past_kv, indices, device):
    """Prune KV cache to keep only tokens at the given indices."""
    indices_t = torch.tensor(sorted(indices), dtype=torch.long, device=device)
    pruned = []
    for layer_kv in past_kv:
        k, v = layer_kv
        pruned.append((k.index_select(2, indices_t), v.index_select(2, indices_t)))
    return tuple(pruned)


def eval_ppl_with_kv(model, input_ids, past_kv, eval_start):
    """Autoregressively evaluate perplexity on held-out tokens using pruned KV cache."""
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
        target = targets[:, i]
        nll = torch.nn.functional.cross_entropy(logits, target).item()
        nlls.append(nll)
        next_token = target.unsqueeze(0)

    return np.exp(np.mean(nlls))


def streaming_llm_indices(seq_len, budget):
    """StreamingLLM: keep first 4 tokens (attention sinks) + most recent (budget-4) tokens."""
    sink_count = min(4, budget)
    recent_count = budget - sink_count
    sinks = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    return sorted(set(sinks + recents))[:budget]


def h2o_indices(attention_weights, budget):
    """H2O: keep tokens that received the highest cumulative attention weight."""
    # attention_weights: list of (batch, heads, seq, seq) tensors, one per layer
    # Average across layers and heads to get a (seq, seq) matrix
    avg_attn = torch.stack([a[0].float().cpu() for a in attention_weights]).mean(dim=(0, 1))
    # Column sum = total attention received by each token position
    token_importance = avg_attn.sum(dim=0)
    _, top_indices = token_importance.topk(min(budget, len(token_importance)))
    return sorted(top_indices.tolist())


def measure_streaming_llm(model, tokenizer, text, budget):
    """StreamingLLM baseline."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 2)
    if eval_start < 10:
        return None

    with torch.no_grad():
        context = input_ids[:, :eval_start - 1]
        output = model(context, use_cache=True)
        past_kv = output.past_key_values

        # Prune: keep sinks + recent
        ctx_len = context.shape[1]
        indices = streaming_llm_indices(ctx_len, min(budget, ctx_len))
        pruned_kv = prune_kv_by_indices(past_kv, indices, DEVICE)

        return eval_ppl_with_kv(model, input_ids, pruned_kv, eval_start)


def measure_h2o(model, tokenizer, text, budget):
    """H2O baseline — needs eager attention to get attention weights."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 2)
    if eval_start < 10:
        return None

    with torch.no_grad():
        context = input_ids[:, :eval_start - 1]
        # H2O needs attention weights to score tokens — this is the reactive cost
        output = model(context, use_cache=True, output_attentions=True)
        past_kv = output.past_key_values

        # Score tokens by cumulative attention, keep heavy hitters
        ctx_len = context.shape[1]
        indices = h2o_indices(output.attentions, min(budget, ctx_len))
        pruned_kv = prune_kv_by_indices(past_kv, indices, DEVICE)

        return eval_ppl_with_kv(model, input_ids, pruned_kv, eval_start)


def main():
    os.makedirs(PHASE3_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    articles = load_articles(split="validation", max_articles=NUM_BENCHMARK_DOCS)

    results = {}
    for budget in KV_BUDGETS:
        # StreamingLLM
        print(f"\n--- StreamingLLM (budget={budget}) ---")
        ppls = []
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for text in tqdm(articles, desc=f"SLLM B={budget}"):
            ppl = measure_streaming_llm(model, tokenizer, text, budget)
            if ppl:
                ppls.append(ppl)
        elapsed = time.time() - start
        vram = torch.cuda.max_memory_allocated() / 1e6
        results[f"streamingllm_{budget}"] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
        print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")

        # H2O
        print(f"\n--- H2O (budget={budget}) ---")
        ppls = []
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for text in tqdm(articles, desc=f"H2O B={budget}"):
            ppl = measure_h2o(model, tokenizer, text, budget)
            if ppl:
                ppls.append(ppl)
        elapsed = time.time() - start
        vram = torch.cuda.max_memory_allocated() / 1e6
        results[f"h2o_{budget}"] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
        print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")

    save_path = os.path.join(PHASE3_DIR, "baseline_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()
