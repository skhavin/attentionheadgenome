# Benchmark on LONG documents (10 articles concatenated per doc, ~4600 tokens).
# This fixes the "everything converges at budget 512" problem.
# Same protocol as benchmark.py: prefill -> prune KV cache -> evaluate held-out tokens.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_concatenated_articles
from phase2.predict_prototypes import predict_prototypes
from phase2.build_retention_mask import predict_retention_mask
from phase3.kv_cache_wrapper import apply_retention_mask
from config import MODEL_NAME, DEVICE, USE_FP16, PHASE3_DIR, PROTOTYPES_PATH, KV_BUDGETS

# Use GPT-2 Medium's full context window
LONG_MAX_SEQ_LEN = 1024  # GPT-2's max position
EVAL_TOKENS = 100        # more eval tokens for long docs
NUM_LONG_DOCS = 20       # 20 concatenated docs


def prune_kv_by_indices(past_kv, indices, device):
    """Prune KV cache to keep only tokens at the given indices."""
    indices_t = torch.tensor(sorted(indices), dtype=torch.long, device=device)
    pruned = []
    for layer_kv in past_kv:
        k, v = layer_kv
        pruned.append((k.index_select(2, indices_t), v.index_select(2, indices_t)))
    return tuple(pruned)


def streaming_llm_indices(seq_len, budget):
    """StreamingLLM: keep first 4 (sinks) + most recent (budget-4) tokens."""
    sink_count = min(4, budget)
    recent_count = budget - sink_count
    sinks = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    return sorted(set(sinks + recents))[:budget]


def h2o_indices(attention_weights, budget):
    """H2O: keep tokens with highest cumulative attention received."""
    avg_attn = torch.stack([a[0].float().cpu() for a in attention_weights]).mean(dim=(0, 1))
    token_importance = avg_attn.sum(dim=0)
    _, top_indices = token_importance.topk(min(budget, len(token_importance)))
    return sorted(top_indices.tolist())


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


def run_method(model, tokenizer, docs, method, budget=None, prototypes=None):
    """Run one method on all docs, return list of PPLs."""
    ppls = []
    for text in tqdm(docs, desc=f"{method} B={budget}"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=LONG_MAX_SEQ_LEN)
        input_ids = tokens["input_ids"].to(DEVICE)
        seq_len = input_ids.shape[1]
        eval_start = seq_len - min(EVAL_TOKENS, seq_len // 4)

        if eval_start < 20:
            continue

        with torch.no_grad():
            context = input_ids[:, :eval_start - 1]
            ctx_len = context.shape[1]

            if method == "full":
                output = model(context, use_cache=True)
                past_kv = output.past_key_values

            elif method == "proactive":
                output = model(context, use_cache=True)
                past_kv = output.past_key_values
                predictions = predict_prototypes(None, prototypes)
                masks = predict_retention_mask(prototypes, predictions, ctx_len, budget)
                past_kv = apply_retention_mask(past_kv, masks, budget, device=DEVICE)

            elif method == "streamingllm":
                output = model(context, use_cache=True)
                past_kv = output.past_key_values
                indices = streaming_llm_indices(ctx_len, min(budget, ctx_len))
                past_kv = prune_kv_by_indices(past_kv, indices, DEVICE)

            elif method == "h2o":
                output = model(context, use_cache=True, output_attentions=True)
                past_kv = output.past_key_values
                indices = h2o_indices(output.attentions, min(budget, ctx_len))
                past_kv = prune_kv_by_indices(past_kv, indices, DEVICE)

            ppl = eval_ppl_with_kv(model, input_ids, past_kv, eval_start)
            if ppl:
                ppls.append(ppl)

    return ppls


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Proactive KV-eviction long-doc benchmark")
    parser.add_argument("--budgets", type=int, nargs="+", default=KV_BUDGETS,
                        help="KV cache budgets to evaluate")
    args = parser.parse_args()
    budgets = args.budgets

    os.makedirs(PHASE3_DIR, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)

    # Load concatenated long docs from validation split
    docs = load_concatenated_articles(split="validation", articles_per_doc=10, max_docs=NUM_LONG_DOCS)

    save_path = os.path.join(PHASE3_DIR, "benchmark_long_results.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results from {save_path}")
    else:
        results = {}

    # Full attention
    if "full" not in results:
        print("\n--- Full Attention (long docs) ---")
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        ppls = run_method(model, tokenizer, docs, "full")
        elapsed = time.time() - start
        vram = torch.cuda.max_memory_allocated() / 1e6
        results["full"] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
        print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")
    else:
        print(f"  [SKIP] full attention already evaluated (PPL={results['full']['ppl']:.2f})")

    # All methods at specified budgets
    for budget in budgets:
        for method in ["proactive", "streamingllm", "h2o"]:
            label = {"proactive": "Proactive", "streamingllm": "StreamingLLM", "h2o": "H2O"}[method]
            print(f"\n--- {label} (budget={budget}, long docs) ---")
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            ppls = run_method(model, tokenizer, docs, method, budget, prototypes)
            elapsed = time.time() - start
            vram = torch.cuda.max_memory_allocated() / 1e6
            key = f"{method}_{budget}"
            results[key] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
            print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {save_path}")

    # Print table
    print("\n=== Long-Document Benchmark Results ===\n")
    print(f"{'Method':<20} {'Budget':>6} {'PPL':>8} {'VRAM(MB)':>9}")
    print("-" * 48)
    print(f"{'Full Attention':<20} {'all':>6} {results['full']['ppl']:>8.2f} {results['full']['vram_mb']:>9.0f}")
    print("-" * 48)
    # Print for all budgets present in the combined results dictionary
    available_budgets = sorted(list(set(
        int(k.split("_")[1]) for k in results.keys() if "_" in k and k.split("_")[1].isdigit()
    )))
    if not available_budgets:
        available_budgets = budgets
    for budget in available_budgets:
        for method, label in [("streamingllm", "StreamingLLM"), ("h2o", "H2O"), ("proactive", "Proactive (ours)")]:
            key = f"{method}_{budget}"
            if key in results:
                r = results[key]
                print(f"{label:<20} {budget:>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f}")
        print("-" * 48)

if __name__ == "__main__":
    main()
