# step1_routing_policy.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Implement the HeadGenome attention compiler routing policy and evaluate
#          its perplexity preservation against StreamingLLM at different cache budgets.
#
# ROUTING POLICY:
#   - Layers with only Sink/Local heads are compressed to [0:4] + last (B-4) tokens.
#   - Layers with Retrieval/Induction heads are preserved at full length.
#
# OUTPUTS:
#   outputs/phase4/routing_policy_results.json

import os
import sys

# Set cache directories BEFORE importing transformers
os.environ["HF_HOME"]             = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"]     = "utf-8"

import time
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR   = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR  = os.path.join(ROOT, "outputs", "phase4")

MODEL_ID   = "unsloth/Llama-3.2-1B"
MODEL_SLUG = "llama-3.2-1b"
K_CLUSTERS = 4
MAX_SEQ_LEN = 512
EVAL_TOKENS = 50
NUM_DOCS    = 15  # Benchmark on 15 long validation docs for speed and reliability


def map_cluster_roles(centroids):
    """
    Map each of the 4 cluster indices to a semantic role:
    sink, local, retrieval, induction.
    Uses std and sink_mass to resolve overlaps.
    """
    n = centroids.shape[0]
    stds = [float(c.std()) for c in centroids]
    sink_masses = [float(c[0:4].sum()) for c in centroids]
    
    # 1. Induction has the absolute lowest std (flattest distribution)
    induction_idx = int(np.argmin(stds))
    
    # 2. Sink has the highest sink mass (early positions)
    sink_idx = int(np.argmax(sink_masses))
    if sink_idx == induction_idx:
        # Fallback: pick the second highest sink mass
        sorted_sink_indices = np.argsort(sink_masses)[::-1]
        sink_idx = int(sorted_sink_indices[1])
        
    # 3. Of the remaining two:
    #    - The one with higher std is retrieval
    #    - The one with lower std is local
    remaining = [i for i in range(n) if i not in (induction_idx, sink_idx)]
    if stds[remaining[0]] > stds[remaining[1]]:
        retrieval_idx = remaining[0]
        local_idx = remaining[1]
    else:
        retrieval_idx = remaining[1]
        local_idx = remaining[0]
        
    return {
        sink_idx: "sink",
        local_idx: "local",
        retrieval_idx: "retrieval",
        induction_idx: "induction"
    }


def get_head_roles():
    json_path = os.path.join(IN_DIR, f"{MODEL_SLUG}_patterns_summary.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] {json_path} missing. Run step3_profile_llama.py first.")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    heads = {}
    for key, hist in data["heads"].items():
        layer, head = map(int, key.split("_"))
        heads[(layer, head)] = np.array(hist, dtype=np.float32)

    keys = sorted(heads.keys())
    X = np.array([heads[k] for k in keys])

    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    role_map = map_cluster_roles(centroids)
    return {k: role_map[labels[i]] for i, k in enumerate(keys)}


def prune_kv_layerwise(past_kv, keep_indices, device):
    """
    Prune DynamicCache layer-by-layer in-place to keep only the specified token indices.
    """
    for layer_idx in range(len(past_kv.layers)):
        k = past_kv.layers[layer_idx].keys
        v = past_kv.layers[layer_idx].values
        indices = torch.tensor(sorted(keep_indices[layer_idx]), dtype=torch.long, device=device)
        
        past_kv.layers[layer_idx].keys = k.index_select(2, indices)
        past_kv.layers[layer_idx].values = v.index_select(2, indices)
        
    return past_kv


def eval_ppl_with_kv(model, input_ids, past_kv, eval_start, keep_indices_fn, budget, device):
    """Autoregressively evaluate perplexity, applying pruning policy after each step."""
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

        # Autoregressive KV update & prune
        seq_len = past_kv.get_seq_length()
        keep_indices = keep_indices_fn(seq_len, budget)
        past_kv = prune_kv_layerwise(past_kv, keep_indices, device)

        next_token = target.unsqueeze(0)

    return np.exp(np.mean(nlls))


def get_streaming_llm_indices(num_layers, seq_len, budget):
    """Uniform StreamingLLM: keep 4 sinks + recent (budget-4) across all layers."""
    sink_count = min(4, budget)
    recent_count = budget - sink_count
    sinks = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    indices = sorted(set(sinks + recents))[:budget]
    return {l: indices for l in range(num_layers)}


def get_headgenome_indices(num_layers, head_roles, seq_len, budget):
    """
    HeadGenome policy:
    - If layer has any Retrieval/Induction head, keep ALL tokens (no eviction).
    - If layer has only Sink/Local heads, keep 4 sinks + last (budget-4) tokens.
    """
    sink_count = min(4, budget)
    recent_count = budget - sink_count
    sinks = list(range(sink_count))
    recents = list(range(max(sink_count, seq_len - recent_count), seq_len))
    compressed_indices = sorted(set(sinks + recents))[:budget]

    keep_indices = {}
    for l in range(num_layers):
        # Check roles of heads in this layer (Llama-3.2-1B has 16 heads)
        layer_roles = [head_roles[(l, h)] for h in range(16)]
        has_retrieval_or_induction = any(r in ["retrieval", "induction"] for r in layer_roles)

        if has_retrieval_or_induction:
            # Keep all tokens in the cache
            keep_indices[l] = list(range(seq_len))
        else:
            # Compress layer using Sink + Local union
            keep_indices[l] = compressed_indices

    return keep_indices


def measure_policy(model, tokenizer, text, budget, policy_name, head_roles, device):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    eval_start = seq_len - min(EVAL_TOKENS, seq_len // 2)
    if eval_start < 10:
        return None

    num_layers = model.config.num_hidden_layers

    # Hook for policy-based index calculation
    if policy_name == "streaming_llm":
        indices_fn = lambda s_len, b: get_streaming_llm_indices(num_layers, s_len, b)
    elif policy_name == "headgenome":
        indices_fn = lambda s_len, b: get_headgenome_indices(num_layers, head_roles, s_len, b)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    with torch.no_grad():
        context = input_ids[:, :eval_start - 1]
        output = model(context, use_cache=True)
        past_kv = output.past_key_values

        # Prune prefill KV cache
        ctx_len = context.shape[1]
        keep_indices = indices_fn(ctx_len, min(budget, ctx_len))
        past_kv = prune_kv_layerwise(past_kv, keep_indices, device)

        return eval_ppl_with_kv(model, input_ids, past_kv, eval_start, indices_fn, budget, device)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading functional head roles...")
    head_roles = get_head_roles()

    print(f"Loading model {MODEL_ID}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    print("Loading validation articles...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
    articles = []
    for item in dataset:
        text = item["text"].strip()
        if len(text.split()) > 150:
            articles.append(text)
            if len(articles) >= NUM_DOCS:
                break

    print(f"Loaded {len(articles)} long articles.")

    budgets = [64, 128, 256]
    results = {}

    for budget in budgets:
        print(f"\nEvaluating budget={budget}...")

        # ── StreamingLLM Baseline ──
        sllm_ppls = []
        start = time.time()
        for text in tqdm(articles, desc=f"SLLM B={budget}"):
            ppl = measure_policy(model, tok, text, budget, "streaming_llm", head_roles, device)
            if ppl:
                sllm_ppls.append(ppl)
        sllm_elapsed = time.time() - start
        sllm_avg_ppl = float(np.mean(sllm_ppls))

        # ── HeadGenome Policy ──
        hg_ppls = []
        start = time.time()
        for text in tqdm(articles, desc=f"HG B={budget}"):
            ppl = measure_policy(model, tok, text, budget, "headgenome", head_roles, device)
            if ppl:
                hg_ppls.append(ppl)
        hg_elapsed = time.time() - start
        hg_avg_ppl = float(np.mean(hg_ppls))

        results[str(budget)] = {
            "streaming_llm": {
                "ppl":      round(sllm_avg_ppl, 4),
                "time_sec": round(sllm_elapsed, 2)
            },
            "headgenome": {
                "ppl":      round(hg_avg_ppl, 4),
                "time_sec": round(hg_elapsed, 2)
            }
        }

        print(f"  StreamingLLM: PPL={sllm_avg_ppl:.2f} ({sllm_elapsed:.1f}s)")
        print(f"  HeadGenome:   PPL={hg_avg_ppl:.2f} ({hg_elapsed:.1f}s)")

    # Save to outputs/phase4/routing_policy_results.json
    out_json = os.path.join(OUT_DIR, "routing_policy_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {out_json}")
    print("\n[DONE] Routing policy evaluation complete.")


if __name__ == "__main__":
    main()
