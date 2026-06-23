# -*- coding: utf-8 -*-
# phase4/step2_cross_arch_eviction.py
#
# PURPOSE: Address Gap 2 - Prove HeadGenome KV eviction generalizes
#          beyond Llama-3.2-1B to GPT-2 Medium and Qwen-2.5-0.5B.
#
# USES:    Entropy-collapse mechanistic labels (not KMeans cluster labels)
#          for routing decisions. This is the correct post-two-axis-methodology approach.
#
# POLICY:  Same as phase4/step1_routing_policy.py:
#   - Layer has Retrieval or Induction head → keep FULL KV cache
#   - Layer has only Sink/Local heads       → compress to [0:4] + recent(budget-4)
#
# OUTPUTS: outputs/phase4/cross_arch_eviction.json

import os
import json
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase4")

NUM_DOCS    = 15
MAX_SEQ_LEN = 512
EVAL_TOKENS = 50
BUDGETS     = [64, 128, 256]

MODELS = [
    {
        "model_id":    "gpt2-medium",
        "slug":        "gpt2",
        "label_file":  "gpt2_retrieval_entropy.json",
        "dtype":       torch.float32,
        "num_heads":   16,
        "num_layers":  24,
    },
    {
        "model_id":    "Qwen/Qwen2.5-0.5B",
        "slug":        "qwen0.5b",
        "label_file":  "qwen0.5b_retrieval_entropy.json",
        "dtype":       torch.bfloat16,
        "num_heads":   14,
        "num_layers":  24,
    },
]


def load_labels(label_file):
    """Load mechanistic labels from the entropy-collapse JSON."""
    path = os.path.join(IN_DIR, label_file)
    if not os.path.exists(path):
        # Try fallback to gpt2_mechanistic_labels.json if target file not found
        if "gpt2" in label_file:
            path = os.path.join(IN_DIR, "gpt2_mechanistic_labels.json")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run the entropy-collapse experiment first.")
        sys.exit(1)
        
    with open(path) as f:
        data = json.load(f)
        
    labels = {}
    # Case 1: Direct "labels" dictionary exists
    if "labels" in data:
        for key, v in data["labels"].items():
            l, h = map(int, key.split("_"))
            labels[(l, h)] = v
        return labels
        
    # Case 2: "heads" exists and maps key to role string directly
    if "heads" in data:
        first_val = list(data["heads"].values())[0]
        if isinstance(first_val, str):
            for key, role in data["heads"].items():
                l, h = map(int, key.split("_"))
                labels[(l, h)] = role
            return labels
        elif isinstance(first_val, dict):
            # Case 3: "heads" maps to detail dict (like gpt2_retrieval_entropy.json), derive roles
            threshold_ret = 0.30
            threshold_ind = -0.50
            threshold_sink = 0.10
            for key, v in data["heads"].items():
                l, h = map(int, key.split("_"))
                delta = v.get("delta")
                me    = v.get("match_entropy")
                nme   = v.get("nonmatch_entropy")
                
                if v.get("nan") or delta is None or me is None:
                    role = "sink"
                elif me < threshold_sink and nme < threshold_sink:
                    role = "sink"
                elif delta > threshold_ret:
                    role = "retrieval"
                elif delta < threshold_ind:
                    role = "induction"
                else:
                    role = "local"
                labels[(l, h)] = role
            return labels

    print(f"[ERROR] Unknown schema in label file {path}")
    sys.exit(1)


def get_streaming_llm_indices(num_layers, seq_len, budget):
    sink_count   = min(4, budget)
    recent_count = budget - sink_count
    sinks   = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    indices = sorted(set(sinks + recents))[:budget]
    return {l: indices for l in range(num_layers)}


def get_headgenome_indices(num_layers, num_heads, head_labels, seq_len, budget):
    sink_count   = min(4, budget)
    recent_count = budget - sink_count
    sinks   = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    compressed = sorted(set(sinks + recents))[:budget]

    keep_indices = {}
    for l in range(num_layers):
        layer_roles = [head_labels.get((l, h), "local") for h in range(num_heads)]
        has_critical = any(r in ("retrieval", "induction") for r in layer_roles)
        keep_indices[l] = list(range(seq_len)) if has_critical else compressed
    return keep_indices


def prune_kv_cache(past_kv, keep_indices, device):
    """Prune DynamicCache layer-by-layer in-place to keep only the specified token indices."""
    for layer_idx in range(len(past_kv.layers)):
        k = past_kv.layers[layer_idx].keys
        v = past_kv.layers[layer_idx].values
        idx = torch.tensor(
            sorted(keep_indices[layer_idx]), dtype=torch.long, device=device
        )
        past_kv.layers[layer_idx].keys = k.index_select(2, idx)
        past_kv.layers[layer_idx].values = v.index_select(2, idx)
    return past_kv


def eval_ppl_with_kv(model, input_ids, past_kv, eval_start, indices_fn, budget, device):
    targets = input_ids[:, eval_start:]
    gen_len = targets.shape[1]
    if gen_len < 5:
        return None
    nlls = []
    next_token = input_ids[:, eval_start - 1:eval_start]

    for i in range(gen_len):
        output = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = output.past_key_values
        logits  = output.logits[:, -1, :]
        target  = targets[:, i]
        nlls.append(torch.nn.functional.cross_entropy(logits, target).item())

        seq_len = past_kv.get_seq_length()
        keep = indices_fn(seq_len, budget)
        past_kv = prune_kv_cache(past_kv, keep, device)
        next_token = target.unsqueeze(0)

    return np.exp(np.mean(nlls))


def measure_policy(model, tokenizer, text, budget, indices_fn, num_layers, device):
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    )
    input_ids = tokens["input_ids"].to(device)
    seq_len   = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 2)
    if eval_start < 10:
        return None

    with torch.no_grad():
        context = input_ids[:, :eval_start - 1]
        output  = model(context, use_cache=True)
        past_kv = output.past_key_values  # tuple of (k, v) per layer

        ctx_len = context.shape[1]
        keep = indices_fn(min(budget, ctx_len), budget)
        past_kv = prune_kv_cache(past_kv, keep, device)
        return eval_ppl_with_kv(
            model, input_ids, past_kv, eval_start, indices_fn, budget, device
        )


def eval_model(cfg, articles, device):
    slug = cfg["slug"]
    print(f"\n{'='*60}")
    print(f"Model: {cfg['model_id']}")
    print(f"{'='*60}")

    head_labels = load_labels(cfg["label_file"])
    num_critical_layers = len(set(
        l for (l, h), role in head_labels.items()
        if role in ("retrieval", "induction")
    ))
    num_layers = cfg["num_layers"]
    num_heads  = cfg["num_heads"]
    print(f"  {num_critical_layers}/{num_layers} layers have retrieval/induction heads (will be preserved)")

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=cfg["dtype"],
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    model_results = {}
    for budget in BUDGETS:
        print(f"\n  Budget={budget}")

        # StreamingLLM
        sllm_fn = lambda sl, b: get_streaming_llm_indices(num_layers, sl, b)
        sllm_ppls, t0 = [], time.time()
        for text in tqdm(articles, desc=f"  SLLM B={budget}"):
            ppl = measure_policy(model, tok, text, budget, sllm_fn, num_layers, device)
            if ppl and ppl < 1e5:
                sllm_ppls.append(ppl)
        sllm_ppl  = float(np.mean(sllm_ppls)) if sllm_ppls else float("nan")
        sllm_time = time.time() - t0

        # HeadGenome
        hg_fn = lambda sl, b: get_headgenome_indices(num_layers, num_heads, head_labels, sl, b)
        hg_ppls, t0 = [], time.time()
        for text in tqdm(articles, desc=f"  HG   B={budget}"):
            ppl = measure_policy(model, tok, text, budget, hg_fn, num_layers, device)
            if ppl and ppl < 1e5:
                hg_ppls.append(ppl)
        hg_ppl  = float(np.mean(hg_ppls)) if hg_ppls else float("nan")
        hg_time = time.time() - t0

        improvement = sllm_ppl / hg_ppl if hg_ppl > 0 else float("nan")
        print(f"    StreamingLLM PPL: {sllm_ppl:.2f}")
        print(f"    HeadGenome PPL:   {hg_ppl:.2f}")
        print(f"    Improvement:      {improvement:.2f}x")

        model_results[str(budget)] = {
            "streaming_llm": {"ppl": round(sllm_ppl, 4), "time_sec": round(sllm_time, 2)},
            "headgenome":    {"ppl": round(hg_ppl, 4),   "time_sec": round(hg_time, 2)},
            "improvement_x": round(improvement, 2),
        }

    del model
    torch.cuda.empty_cache()
    return model_results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading WikiText-103 validation articles...")
    dataset  = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
    articles = [item["text"].strip() for item in dataset if len(item["text"].split()) > 150]
    articles = articles[:NUM_DOCS]
    print(f"Loaded {len(articles)} articles.")

    all_results = {}
    for cfg in MODELS:
        all_results[cfg["slug"]] = eval_model(cfg, articles, device)

    # Cross-model summary
    print(f"\n{'='*70}")
    print("CROSS-ARCHITECTURE KV EVICTION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<14}  {'Budget':<8}  {'SLLM PPL':>10}  {'HG PPL':>8}  {'Improvement':>12}")
    print(f"  {'-'*56}")
    for slug, res in all_results.items():
        for budget_str, vals in res.items():
            sllm = vals["streaming_llm"]["ppl"]
            hg   = vals["headgenome"]["ppl"]
            imp  = vals["improvement_x"]
            print(f"  {slug:<14}  {budget_str:<8}  {sllm:>10.2f}  {hg:>8.2f}  {imp:>11.2f}x")

    out_path = os.path.join(OUT_DIR, "cross_arch_eviction.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
