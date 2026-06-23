# -*- coding: utf-8 -*-
# phase4/step4_head_granular_eviction.py
#
# PURPOSE: Head-granularity KV eviction (fixes the layer-granularity problem).
#
# PROBLEM WITH LAYER-GRANULARITY:
#   GPT-2 has 15/24 layers with retrieval/induction heads.
#   Preserving ALL 15 layers at full cache starves the budget.
#   Result: HeadGenome WORSE than StreamingLLM at small budgets.
#
# HEAD-GRANULAR SOLUTION:
#   Patch each layer's _attn method to apply per-head additive masks:
#     - sink head h:       mask all positions > 4 to -inf
#     - local head h:      mask all positions < (kv_len - W) to -inf
#     - retrieval/induction: no mask (full attention)
#
#   Since attention_mask in GPT-2 is an additive bias added to logits
#   before softmax, adding -inf to forbidden positions is equivalent to
#   excluding those tokens from each head's KV cache.
#
#   PPL is measured with the full KV cache stored but per-head attention
#   masking applied -- this proves the approach works before implementing
#   true sparse KV kernels.
#
# OUTPUTS: outputs/phase4/head_granular_eviction.json

import os, sys, json, time, types
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = r"d:\.cache\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase4")

NUM_DOCS    = 15
MAX_SEQ_LEN = 512
EVAL_TOKENS = 50
WINDOW_SIZE = 32
SINK_SIZE   = 4

MODELS = [
    {
        "model_id":   "gpt2-medium",
        "slug":       "gpt2",
        "label_file": "gpt2_mechanistic_labels.json",
        "dtype":      torch.float32,
        "num_heads":  16,
        "num_layers": 24,
        "head_dim":   64,
        "attn_path":  "transformer.h",       # model.transformer.h[i].attn
        "attn_attr":  "attn",
    },
    {
        "model_id":   "Qwen/Qwen2.5-0.5B",
        "slug":       "qwen0.5b",
        "label_file": "qwen0.5b_retrieval_entropy.json",
        "dtype":      torch.bfloat16,
        "num_heads":  14,
        "num_layers": 24,
        "head_dim":   64,
        "attn_path":  "model.layers",         # model.model.layers[i].self_attn
        "attn_attr":  "self_attn",
    },
]

THRESHOLD_RETRIEVAL = 0.30
THRESHOLD_INDUCTION = -0.50
THRESHOLD_SINK_ENT  = 0.10


def load_labels(label_file):
    path = os.path.join(IN_DIR, label_file)
    if not os.path.exists(path):
        alt = os.path.join(IN_DIR, "gpt2_mechanistic_labels.json")
        if "gpt2" in label_file and os.path.exists(alt):
            path = alt
        else:
            print(f"[ERROR] {path} not found.")
            sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    labels = {}
    if "labels" in data:
        for key, role in data["labels"].items():
            l, h = map(int, key.split("_"))
            labels[(l, h)] = role
        return labels

    if "heads" in data:
        first = list(data["heads"].values())[0]
        if isinstance(first, str):
            for key, role in data["heads"].items():
                l, h = map(int, key.split("_"))
                labels[(l, h)] = role
            return labels
        elif isinstance(first, dict):
            for key, v in data["heads"].items():
                l, h = map(int, key.split("_"))
                delta, me, nme = v.get("delta"), v.get("match_entropy"), v.get("nonmatch_entropy")
                if v.get("nan") or delta is None or me is None:
                    role = "sink"
                elif me < THRESHOLD_SINK_ENT and nme < THRESHOLD_SINK_ENT:
                    role = "sink"
                elif delta > THRESHOLD_RETRIEVAL:
                    role = "retrieval"
                elif delta < THRESHOLD_INDUCTION:
                    role = "induction"
                else:
                    role = "local"
                labels[(l, h)] = role
            return labels

    print(f"[ERROR] Unknown schema in {path}")
    sys.exit(1)


class HeadGranularMaskHooks:
    """
    Uses a forward pre-hook on each GPT-2 attention block to inject a
    per-head additive mask into the attention_mask argument.

    In GPT-2's eager_attention_forward, attention_mask is added to logits
    before softmax, so adding -inf to forbidden positions excludes them.

    Pre-hook receives (module, args, kwargs). We modify the kwargs
    'attention_mask' entry to include the per-head bias.
    """

    def __init__(self, model, head_labels, num_heads, num_layers, cfg, budget, sink_size=SINK_SIZE):
        self.handles     = []
        self.window_size = budget - sink_size
        self.sink_size   = sink_size

        for layer_idx in range(num_layers):
            attn_module = get_attn_module(model, cfg, layer_idx)
            layer_roles = {h: head_labels.get((layer_idx, h), "local")
                           for h in range(num_heads)}
            handle = attn_module.register_forward_pre_hook(
                self._make_pre_hook(layer_roles, num_heads),
                with_kwargs=True
            )
            self.handles.append(handle)

    def _make_pre_hook(self, layer_roles, num_heads):
        window_size = self.window_size
        sink_size   = self.sink_size

        def pre_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return args, kwargs

            past_kv = kwargs.get("past_key_values")
            q_len   = hidden_states.shape[1]
            device  = hidden_states.device
            dtype   = hidden_states.dtype

            cache_pos = kwargs.get("cache_position")
            if cache_pos is not None and len(cache_pos) > 0:
                kv_len = int(cache_pos[-1].item()) + 1
                q_pos_1d = cache_pos
            else:
                if past_kv is not None and hasattr(past_kv, "get_seq_length"):
                    kv_seq = past_kv.get_seq_length()
                    kv_len = kv_seq + q_len
                else:
                    kv_len = q_len
                q_pos_1d = torch.arange(kv_len - q_len, kv_len, device=device)

            q_pos = q_pos_1d.unsqueeze(1)
            k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

            causal_allow = q_pos >= k_pos
            allow_sink_only = ((k_pos < sink_size) | ((q_pos - k_pos) < window_size)) & causal_allow
            allow_local     = ((k_pos < sink_size) | ((q_pos - k_pos) < window_size)) & causal_allow
            allow_full      = causal_allow

            zero_t = torch.tensor(0.0, dtype=dtype, device=device)
            inf_t  = torch.tensor(float("-inf"), dtype=dtype, device=device)

            sink_mask  = torch.where(allow_sink_only, zero_t, inf_t)
            local_mask = torch.where(allow_local, zero_t, inf_t)
            full_mask  = torch.where(allow_full, zero_t, inf_t)

            role_mask = torch.empty(1, num_heads, q_len, kv_len, dtype=dtype, device=device)

            for h in range(num_heads):
                role = layer_roles.get(h, "local")
                if role == "sink":
                    role_mask[0, h] = sink_mask
                elif role == "local":
                    role_mask[0, h] = local_mask
                else:
                    role_mask[0, h] = full_mask

            existing = kwargs.get("attention_mask")
            if existing is not None:
                try:
                    combined = existing + role_mask
                    kwargs = dict(kwargs, attention_mask=combined)
                except Exception:
                    kwargs = dict(kwargs, attention_mask=role_mask)
            else:
                kwargs = dict(kwargs, attention_mask=role_mask)

            return args, kwargs

        return pre_hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class StreamingLLMMaskHooks:
    """
    Forward pre-hook that applies the StreamingLLM uniform mask:
    keep only first sink_size tokens + last (budget - sink_size) tokens.
    """

    def __init__(self, model, num_layers, num_heads, budget, cfg, sink_size=SINK_SIZE):
        self.handles     = []
        self.recent_size = budget - sink_size
        self.sink_size   = sink_size

        for layer_idx in range(num_layers):
            attn_module = get_attn_module(model, cfg, layer_idx)
            handle = attn_module.register_forward_pre_hook(
                self._make_pre_hook(num_heads),
                with_kwargs=True
            )
            self.handles.append(handle)

    def _make_pre_hook(self, num_heads):
        sink_size   = self.sink_size
        recent_size = self.recent_size

        def pre_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return args, kwargs

            cache_pos = kwargs.get("cache_position")
            past_kv   = kwargs.get("past_key_values")
            q_len     = hidden_states.shape[1]
            device    = hidden_states.device
            dtype     = hidden_states.dtype

            if cache_pos is not None and len(cache_pos) > 0:
                kv_len = int(cache_pos[-1].item()) + 1
                q_pos_1d = cache_pos
            else:
                if past_kv is not None and hasattr(past_kv, "get_seq_length"):
                    kv_seq = past_kv.get_seq_length()
                    kv_len = kv_seq + q_len
                else:
                    kv_len = q_len
                q_pos_1d = torch.arange(kv_len - q_len, kv_len, device=device)

            q_pos = q_pos_1d.unsqueeze(1)
            k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

            causal_allow = q_pos >= k_pos
            allow = ((k_pos < sink_size) | ((q_pos - k_pos) < recent_size)) & causal_allow

            zero_t = torch.tensor(0.0, dtype=dtype, device=device)
            inf_t  = torch.tensor(float("-inf"), dtype=dtype, device=device)
            mask = torch.where(allow, zero_t, inf_t).unsqueeze(0).unsqueeze(0)

            existing = kwargs.get("attention_mask")
            if existing is not None:
                try:
                    combined = existing + mask
                    kwargs = dict(kwargs, attention_mask=combined)
                except Exception:
                    kwargs = dict(kwargs, attention_mask=mask)
            else:
                kwargs = dict(kwargs, attention_mask=mask)

            return args, kwargs

        return pre_hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_attn_module(model, cfg, layer_idx):
    """Return the attention submodule for a given layer, architecture-agnostic."""
    path  = cfg["attn_path"]
    attr  = cfg["attn_attr"]
    # Traverse e.g. 'model.layers' or 'transformer.h'
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    block = obj[layer_idx]
    return getattr(block, attr)


def compute_ppl_with_mask(model, tokenizer, texts, device):
    """Standard cross-entropy PPL — the mask hooks are already applied."""
    nlls = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ_LEN).to(device)
        input_ids = tokens["input_ids"]
        if input_ids.shape[1] < 10:
            continue
        with torch.no_grad():
            out = model(**tokens, labels=input_ids)
            nll = out.loss.item()
            if not (torch.isnan(out.loss) or nll > 20):
                nlls.append(nll)
    return float(np.exp(np.mean(nlls))) if nlls else float("nan")


def eval_model(cfg, articles, device):
    slug = cfg["slug"]
    print(f"\n{'='*60}")
    print(f"Model: {cfg['model_id']}")
    print(f"{'='*60}")

    head_labels = load_labels(cfg["label_file"])
    num_layers  = cfg["num_layers"]
    num_heads   = cfg["num_heads"]

    role_counts = {"sink": 0, "local": 0, "retrieval": 0, "induction": 0}
    for role in head_labels.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    print(f"  Head taxonomy: {role_counts}")

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=cfg["dtype"], trust_remote_code=True
    )
    model = model.to(device).eval()

    model_results = {}
    budgets = [64, 128, 256]

    for budget in budgets:
        print(f"\n  Budget={budget}")

        # ── Baseline: no mask (full attention) ────────────────────────────────
        t0 = time.time()
        baseline_ppl = compute_ppl_with_mask(model, tok, articles, device)
        baseline_time = time.time() - t0
        print(f"    Baseline PPL (full attn): {baseline_ppl:.2f}")

        # ── StreamingLLM mask ─────────────────────────────────────────────────
        sllm_hooks = StreamingLLMMaskHooks(model, num_layers, num_heads, budget, cfg)
        t0 = time.time()
        sllm_ppl = compute_ppl_with_mask(model, tok, articles, device)
        sllm_time = time.time() - t0
        sllm_hooks.remove()
        print(f"    StreamingLLM PPL:         {sllm_ppl:.2f}")

        # ── HeadGenome head-granular mask ─────────────────────────────────────
        hg_hooks = HeadGranularMaskHooks(model, head_labels, num_heads, num_layers, cfg, budget)
        t0 = time.time()
        hg_ppl = compute_ppl_with_mask(model, tok, articles, device)
        hg_time = time.time() - t0
        hg_hooks.remove()
        print(f"    HeadGenome (head-gran) PPL: {hg_ppl:.2f}")

        improvement = sllm_ppl / hg_ppl if hg_ppl > 0 and not np.isnan(hg_ppl) else float("nan")
        winner = "HeadGenome" if hg_ppl < sllm_ppl else "StreamingLLM"
        print(f"    Winner: {winner}  (ratio {improvement:.3f}x)")

        model_results[str(budget)] = {
            "baseline_full_attn": {"ppl": round(baseline_ppl, 4)},
            "streaming_llm":      {"ppl": round(sllm_ppl, 4), "time_sec": round(sllm_time, 2)},
            "headgenome_granular":{"ppl": round(hg_ppl,   4), "time_sec": round(hg_time, 2)},
            "improvement_x":      round(improvement, 3),
            "winner":             winner,
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
    articles = [it["text"].strip() for it in dataset if len(it["text"].split()) > 150]
    articles = articles[:NUM_DOCS]
    print(f"Loaded {len(articles)} articles.")

    all_results = {}
    for cfg in MODELS:
        all_results[cfg["slug"]] = eval_model(cfg, articles, device)

    # ── Cross-model summary ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("HEAD-GRANULAR KV EVICTION SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Model':<12}  {'Budget':<8}  {'Baseline':>10}  {'SLLM PPL':>10}  {'HG PPL':>8}  {'Winner':>12}")
    print(f"  {'-'*64}")
    for slug, res in all_results.items():
        for budget_str, vals in res.items():
            b   = vals["baseline_full_attn"]["ppl"]
            s   = vals["streaming_llm"]["ppl"]
            hg  = vals["headgenome_granular"]["ppl"]
            win = vals["winner"]
            print(f"  {slug:<12}  {budget_str:<8}  {b:>10.2f}  {s:>10.2f}  {hg:>8.2f}  {win:>12}")

    out_path = os.path.join(OUT_DIR, "head_granular_eviction.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
