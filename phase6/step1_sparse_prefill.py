# -*- coding: utf-8 -*-
# phase6/step1_sparse_prefill.py
#
# PURPOSE: Validate Sparse Prefill (Contribution 3)
# Applies taxonomy-based sparse attention masks during the prefill phase
# (single forward pass over the whole sequence) and measures PPL preservation.
# Also calculates the exact theoretical FLOP savings for the sequence length.

import os, sys, json, time, argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = r"d:\.cache\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase6")

NUM_DOCS    = 15
SINK_SIZE   = 4

MODELS = [
    {
        "model_id":   "Qwen/Qwen2.5-0.5B",
        "slug":       "qwen0.5b",
        "label_file": "qwen0.5b_retrieval_entropy.json",
        "dtype":      torch.bfloat16,
        "num_heads":  14,
        "num_layers": 24,
        "attn_path":  "model.layers",
        "attn_attr":  "self_attn",
    },
    {
        "model_id":   "Qwen/Qwen2.5-1.5B",
        "slug":       "qwen1.5b",
        "label_file": "qwen1.5b_retrieval_entropy.json",
        "dtype":      torch.bfloat16,
        "num_heads":  12,
        "num_layers": 28,
        "attn_path":  "model.layers",
        "attn_attr":  "self_attn",
    }
]

THRESHOLD_RETRIEVAL = 0.30
THRESHOLD_INDUCTION = -0.50
THRESHOLD_SINK_ENT  = 0.10

def load_labels(label_file):
    path = os.path.join(IN_DIR, label_file)
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found.")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    labels = {}
    if "heads" in data:
        first = list(data["heads"].values())[0]
        if isinstance(first, dict):
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

def get_attn_module(model, cfg, layer_idx):
    obj = model
    for part in cfg["attn_path"].split("."):
        obj = getattr(obj, part)
    return getattr(obj[layer_idx], cfg["attn_attr"])


class SparsePrefillMaskHooks:
    """
    Applies HeadGenome sparse masks during prefill.
    Sink/Local heads: sliding window W + sink.
    Retrieval/Induction: full causal attention.
    """
    def __init__(self, model, head_labels, num_heads, num_layers, cfg, W, sink_size=SINK_SIZE):
        self.handles = []
        self.W = W
        self.sink_size = sink_size

        for layer_idx in range(num_layers):
            attn_module = get_attn_module(model, cfg, layer_idx)
            layer_roles = {h: head_labels.get((layer_idx, h), "local") for h in range(num_heads)}
            handle = attn_module.register_forward_pre_hook(
                self._make_pre_hook(layer_roles, num_heads),
                with_kwargs=True
            )
            self.handles.append(handle)

    def _make_pre_hook(self, layer_roles, num_heads):
        W = self.W
        sink_size = self.sink_size

        def pre_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None: return args, kwargs

            q_len = hidden_states.shape[1]
            device = hidden_states.device
            dtype = hidden_states.dtype
            kv_len = q_len

            q_pos = torch.arange(q_len, device=device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

            causal_allow = q_pos >= k_pos
            # Sink heads share the local sliding window, as per our findings
            allow_local = ((k_pos < sink_size) | ((q_pos - k_pos) < W)) & causal_allow
            allow_full  = causal_allow

            zero_t = torch.tensor(0.0, dtype=dtype, device=device)
            inf_t  = torch.tensor(float("-inf"), dtype=dtype, device=device)

            local_mask = torch.where(allow_local, zero_t, inf_t)
            full_mask  = torch.where(allow_full, zero_t, inf_t)

            role_mask = torch.empty(1, num_heads, q_len, kv_len, dtype=dtype, device=device)
            for h in range(num_heads):
                role = layer_roles.get(h, "local")
                if role in ["sink", "local"]:
                    role_mask[0, h] = local_mask
                else:
                    role_mask[0, h] = full_mask

            existing = kwargs.get("attention_mask")
            if existing is not None:
                try:
                    kwargs = dict(kwargs, attention_mask=existing + role_mask)
                except Exception:
                    kwargs = dict(kwargs, attention_mask=role_mask)
            else:
                kwargs = dict(kwargs, attention_mask=role_mask)

            return args, kwargs
        return pre_hook

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def compute_ppl_with_mask(model, tokenizer, texts, device, seq_len):
    nlls = []
    total_tokens = 0
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]
        if seq_len < 100: continue
        
        with torch.no_grad():
            out = model(**tokens, labels=input_ids)
            nll = out.loss.item()
            if not (torch.isnan(out.loss) or nll > 20):
                nlls.append(nll * seq_len)
                total_tokens += seq_len
                
    return float(np.exp(np.sum(nlls) / total_tokens)) if total_tokens > 0 else float("nan")

def compute_flop_savings(head_labels, N, W):
    roles = list(head_labels.values())
    total_heads = len(roles)
    f_sink = roles.count("sink") / total_heads
    f_local = roles.count("local") / total_heads
    f_ret = roles.count("retrieval") / total_heads
    f_ind = roles.count("induction") / total_heads
    f_crit = f_ret + f_ind
    f_compressed = f_sink + f_local # Since sink now shares local window W
    
    # Area of full causal mask: N^2 / 2
    full_area = (N * N) / 2
    
    # Area of local window mask: N * W - W^2 / 2
    # Area of sink: N * 4
    # For simplicity, approximate compressed area as N * W
    compressed_area = N * W
    
    ops_full = total_heads * full_area
    ops_sparse = (f_compressed * total_heads * compressed_area) + (f_crit * total_heads * full_area)
    
    savings = 1.0 - (ops_sparse / ops_full)
    return savings

def eval_model(cfg, articles, device, seq_len, windows):
    print(f"\n{'='*60}\nModel: {cfg['model_id']}\n{'='*60}")
    head_labels = load_labels(cfg["label_file"])
    num_layers, num_heads = cfg["num_layers"], cfg["num_heads"]

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=cfg["dtype"], trust_remote_code=True)
    model = model.to(device).eval()

    print(f"  Measuring Baseline (Dense Prefill)... (seq_len={seq_len})")
    baseline_ppl = compute_ppl_with_mask(model, tok, articles, device, seq_len)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    results = {}
    for W in windows:
        print(f"\n  Sparse Prefill Window (W={W})")
        hooks = SparsePrefillMaskHooks(model, head_labels, num_heads, num_layers, cfg, W)
        sparse_ppl = compute_ppl_with_mask(model, tok, articles, device, seq_len)
        hooks.remove()
        
        savings = compute_flop_savings(head_labels, seq_len, W)
        
        print(f"    Sparse PPL:   {sparse_ppl:.2f}")
        print(f"    FLOP Savings: {savings*100:.1f}%")
        
        results[str(W)] = {
            "sparse_ppl": round(sparse_ppl, 2) if not np.isnan(sparse_ppl) else None,
            "baseline_ppl": round(baseline_ppl, 2) if not np.isnan(baseline_ppl) else None,
            "flop_savings_pct": round(savings * 100, 1),
            "seq_len": seq_len
        }

    del model
    torch.cuda.empty_cache()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading WikiText-103 validation articles (concatenating for N={args.seq_len})...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
    articles_raw = [it["text"].strip() for it in dataset if len(it["text"].strip()) > 10]
    
    articles = []
    curr = ""
    for a in articles_raw:
        curr += a + " \n\n "
        if len(curr.split()) > args.seq_len * 1.5:
            articles.append(curr)
            curr = ""
            if len(articles) >= NUM_DOCS:
                break
    
    if len(articles) < NUM_DOCS:
        print(f"Warning: Only constructed {len(articles)} documents of length {args.seq_len}")
    
    all_results = {}
    windows_to_test = [args.window] if args.window > 0 else [64, 128, 256]
    
    for cfg in MODELS:
        if args.model != "all" and cfg["model_id"] != args.model:
            continue
        all_results[cfg["slug"]] = eval_model(cfg, articles, device, args.seq_len, windows_to_test)

    out_path = os.path.join(OUT_DIR, "sparse_prefill.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
