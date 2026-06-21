# PG-19 benchmark — long books benchmark on GPT-2 Medium.
# Each book is ~70k tokens. We use 1024-token chunks (GPT-2 max position).
# Tests proactive vs StreamingLLM vs H2O on real long-form text.

import sys, os
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from phase2.predict_prototypes import predict_prototypes
from phase2.build_retention_mask import predict_retention_mask
from phase3.kv_cache_wrapper import apply_retention_mask
from config import MODEL_NAME, DEVICE, USE_FP16, PROTOTYPES_PATH, KV_BUDGETS

PG19_SEQ_LEN = 1024     # GPT-2 max position
EVAL_TOKENS = 100        # tokens to evaluate per chunk
NUM_BOOKS = 10           # books to use
CHUNKS_PER_BOOK = 5      # sample 5 non-overlapping chunks from each book

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase5")


def streaming_llm_indices(seq_len, budget):
    sink_count = min(4, budget)
    recent_count = budget - sink_count
    sinks = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    return sorted(set(sinks + recents))[:budget]


def h2o_indices(attention_weights, budget):
    avg_attn = torch.stack([a[0].float().cpu() for a in attention_weights]).mean(dim=(0, 1))
    token_importance = avg_attn.sum(dim=0)
    _, top_indices = token_importance.topk(min(budget, len(token_importance)))
    return sorted(top_indices.tolist())


def prune_kv(past_kv, indices, device):
    indices_t = torch.tensor(indices, dtype=torch.long, device=device)
    pruned = []
    for layer_kv in past_kv:
        k, v = layer_kv
        pruned.append((k.index_select(2, indices_t), v.index_select(2, indices_t)))
    return tuple(pruned)


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
        target = targets[:, i]
        nll = torch.nn.functional.cross_entropy(logits, target).item()
        nlls.append(nll)
        next_token = target.unsqueeze(0)

    return np.exp(np.mean(nlls))


def extract_chunks(text, tokenizer, seq_len, num_chunks):
    """Extract non-overlapping chunks of seq_len tokens from a book."""
    all_tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    total = len(all_tokens)

    chunks = []
    if total < seq_len:
        return chunks

    # Evenly space chunks through the book
    step = max(seq_len, (total - seq_len) // num_chunks)
    for start in range(0, total - seq_len, step):
        if len(chunks) >= num_chunks:
            break
        chunk = all_tokens[start:start + seq_len].unsqueeze(0)
        chunks.append(chunk)

    return chunks


def run_one_chunk(model, input_ids, method, budget, prototypes=None):
    """Run one method on one chunk, return PPL."""
    seq_len = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 4)
    if eval_start < 20:
        return None

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
            past_kv = prune_kv(past_kv, indices, DEVICE)

        elif method == "h2o":
            output = model(context, use_cache=True, output_attentions=True)
            past_kv = output.past_key_values
            indices = h2o_indices(output.attentions, min(budget, ctx_len))
            past_kv = prune_kv(past_kv, indices, DEVICE)

        return eval_ppl(model, input_ids, past_kv, eval_start)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading GPT-2 Medium...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)

    # Load PG-19 books (using emozilla/pg19 Parquet version to bypass Hugging Face script security blocking)
    print("Loading PG-19 dataset (streaming mode)...")
    ds = load_dataset("emozilla/pg19", split="test", streaming=True).take(NUM_BOOKS)

    # Extract chunks from each book
    all_chunks = []
    for i, example in enumerate(ds):
        text = example["text"]
        book_chunks = extract_chunks(text, tokenizer, PG19_SEQ_LEN, CHUNKS_PER_BOOK)
        all_chunks.extend([(c.to(DEVICE), i) for c in book_chunks])
        print(f"  Book {i}: {len(text)} chars, {len(book_chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    results = {}

    # Full attention
    print("\n--- Full Attention (PG-19) ---")
    ppls = []
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for chunk, _ in tqdm(all_chunks, desc="Full attn"):
        ppl = run_one_chunk(model, chunk, "full", None, None)
        if ppl:
            ppls.append(ppl)
    elapsed = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1e6
    results["full"] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
    print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")

    # All methods at all budgets
    for budget in KV_BUDGETS:
        for method in ["proactive", "streamingllm", "h2o"]:
            label = {"proactive": "Proactive (ours)", "streamingllm": "StreamingLLM", "h2o": "H2O"}[method]
            print(f"\n--- {label} (budget={budget}, PG-19) ---")
            ppls = []
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            for chunk, _ in tqdm(all_chunks, desc=f"{method} B={budget}"):
                ppl = run_one_chunk(model, chunk, method, budget, prototypes)
                if ppl:
                    ppls.append(ppl)
            elapsed = time.time() - start
            vram = torch.cuda.max_memory_allocated() / 1e6
            key = f"{method}_{budget}"
            results[key] = {"ppl": np.mean(ppls), "vram_mb": vram, "time_s": elapsed}
            print(f"  PPL={np.mean(ppls):.2f}, VRAM={vram:.0f}MB, Time={elapsed:.1f}s")

    # Save
    save_path = os.path.join(OUTPUT_DIR, "pg19_benchmark.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {save_path}")

    # Print table
    print("\n=== PG-19 Benchmark Results ===\n")
    print(f"{'Method':<20} {'Budget':>6} {'PPL':>8} {'VRAM(MB)':>9} {'Time(s)':>8}")
    print("-" * 56)
    print(f"{'Full Attention':<20} {'all':>6} {results['full']['ppl']:>8.2f} {results['full']['vram_mb']:>9.0f} {results['full']['time_s']:>8.1f}")
    for budget in KV_BUDGETS:
        print("-" * 56)
        for method, label in [("streamingllm", "StreamingLLM"), ("h2o", "H2O"), ("proactive", "Proactive (ours)")]:
            key = f"{method}_{budget}"
            if key in results:
                r = results[key]
                print(f"{label:<20} {budget:>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")


if __name__ == "__main__":
    main()
