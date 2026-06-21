# phase7/audit_heads.py
#
# Phase 1 — Per-Head Error Audit (L∞ / KL Table)
#
# For every (layer, head), computes:
#   1. L∞ between closed-form substitution output and full softmax output
#   2. KL divergence on downstream logits after substituting that single head
#
# across three prompt categories:
#   A. WikiText-103 held-out test (natural distribution)
#   B. Copy-trigger prompts (varying repeat distance and n-gram length)
#   C. Needle-in-haystack prompts (key fact at N/4, N/2, 3N/4)
#
# Tier classification (both L∞ AND logit-KL must pass for Tier 1):
#   Tier 1 — L∞ < 0.001 AND kl_approx < 0.01 on BOTH natural + copy-trigger
#             → safe closed-form substitution
#   Tier 2 — L∞ < 0.001 on natural, > 0.01 on copy-trigger
#             OR kl_approx spikes on copy-trigger (> KL threshold)
#             → regime-switching: use runtime detector
#   Tier 3 — everything else → full attention required
#
# Output: outputs/phase7/head_audit.json (human-readable) +
#         outputs/phase7/head_audit.pkl  (full tensors for downstream scripts)
#
# Usage:
#   python phase7/audit_heads.py
#   python phase7/audit_heads.py --model gpt2-medium --num_natural 200 --num_copy 100 --num_niah 50

import sys, os, argparse, json, pickle, math
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PHASE7_DIR, DATASET_NAME, DATASET_CONFIG
from phase7.substitutes import sink_substitute, local_substitute

# ---------------------------------------------------------------------------
# Thresholds for tier classification — differentiated by head type
# ---------------------------------------------------------------------------
# Sink heads: attn weights concentrate on first K tokens → check attn-weight L∞.
# Local heads: attn weights are non-uniform over window → skip attn L∞, check
#              absolute output L∞ (pre-o_proj) and scaled KL instead.

TIER1_SINK_ATTN_LINF  = 0.05   # attention-weight L∞ vs implied sink pattern
TIER1_SINK_KL_MAX     = 0.010  # projected, RMSNorm-scaled KL proxy

TIER1_LOCAL_OUT_LINF  = 0.15   # absolute L∞ on per-head value output (pre-o_proj)
TIER1_LOCAL_KL_MAX    = 0.010  # projected, RMSNorm-scaled KL proxy

# Legacy fallback for "sink_and_local" mode (head type unknown)
TIER1_ATTN_LINF   = 0.05
TIER1_OUTPUT_LINF = 0.15
TIER1_KL_MAX      = 0.010


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — Per-head L∞/KL audit")
    p.add_argument("--model", default="gpt2-medium",
                   help="HuggingFace model id (default: gpt2-medium)")
    p.add_argument("--num_natural", type=int, default=200,
                   help="Number of WikiText-103 test documents (default: 200)")
    p.add_argument("--num_copy", type=int, default=100,
                   help="Number of copy-trigger prompts (default: 100)")
    p.add_argument("--num_niah", type=int, default=50,
                   help="Number of needle-in-haystack prompts (default: 50)")
    p.add_argument("--seq_len", type=int, default=512,
                   help="Sequence length for prompts (default: 512)")
    p.add_argument("--head_type_map", type=str, default=None,
                   help="Path to JSON mapping 'layer,head' -> type. "
                        "If None, all heads are audited as both sink and local.")
    p.add_argument("--num_sink_tokens", type=int, default=4)
    p.add_argument("--local_window", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-prefix", default="",
                   help="Prefix for output files (e.g., 'qwen_' to get qwen_head_audit.json)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device):
    """
    Load model for auditing.  Does NOT pass attn_implementation — GPT-2 does
    not support that kwarg (only LLaMA/Qwen do).  output_attentions=True is
    passed at inference time instead, which works on all architectures.
    """
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config._attn_implementation = "eager"

    OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")
    strategies = [
        ({"device_map": {"": str(device)}}, "single-device"),
        ({"device_map": "auto"},             "auto"),
        ({"device_map": "auto",
          "offload_folder": OFFLOAD_DIR,
          "offload_state_dict": True},       "disk-offload"),
    ]
    for extra, tag in strategies:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **extra,
            )
            model.eval()
            print(f"  Loaded [{tag}]")
            return model, tokenizer
        except Exception as e:
            print(f"  Failed [{tag}]: {str(e)[:120]}")
    raise RuntimeError(f"Could not load {model_name}")


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def detect_arch(model) -> str:
    cls = type(model).__name__.lower()
    if "gpt2" in cls:
        return "gpt2"
    elif "llama" in cls or "qwen" in cls or "mistral" in cls:
        return "llama"
    return "generic"


def iter_attn_layers(model, arch: str):
    """Yield (layer_idx, attn_module) in order."""
    if arch == "gpt2":
        for i, block in enumerate(model.transformer.h):
            yield i, block.attn
    elif arch == "llama":
        for i, layer in enumerate(model.model.layers):
            yield i, layer.self_attn
    else:
        # Generic: walk named modules and pick anything with q_proj or c_attn
        seen = set()
        idx = 0
        for name, mod in model.named_modules():
            if (hasattr(mod, "c_attn") or hasattr(mod, "q_proj")) and id(mod) not in seen:
                seen.add(id(mod))
                yield idx, mod
                idx += 1


def get_head_o_proj_weight(
    attn_module, arch: str, h_idx: int, d_head: int
) -> "torch.Tensor | None":
    """
    Return the output-projection weight slice for head h_idx.
    Shape: [d_head, d_model] — ready for: diff_head @ w → [*, d_model]

    Handles architectural differences:
      GPT-2  : uses Conv1D  → weight.shape = [d_in, d_out] = [n_heads*d_head, d_model]
      LLaMA/Qwen : uses nn.Linear → weight.shape = [d_out, d_in] = [d_model, n_heads*d_head]
    """
    if arch == "gpt2":
        # Conv1D: weight[i,j] = input-i → output-j, so weight.shape = [d_in, d_model]
        c_proj = getattr(attn_module, "c_proj", None)
        if c_proj is None:
            return None
        w = c_proj.weight   # [n_heads*d_head, d_model]
        return w[h_idx * d_head : (h_idx + 1) * d_head, :]   # [d_head, d_model]
    else:
        o_proj = getattr(attn_module, "o_proj", None)
        if o_proj is None:
            return None
        # nn.Linear: weight.shape = [d_model, n_heads*d_head]
        w = o_proj.weight   # [d_model, n_heads*d_head]
        return w[:, h_idx * d_head : (h_idx + 1) * d_head].t()  # [d_head, d_model]


# ---------------------------------------------------------------------------
# V extraction — forward hook (NOT pre-hook)
# ---------------------------------------------------------------------------

def extract_all_V(model, input_ids, arch: str, device: torch.device):
    """
    Register a forward hook on EVERY attention layer's value projection submodule
    to capture the V projection for ALL heads in a single forward pass.
    """
    V_captured = {}
    handles = []

    def make_hook(l_idx, attn_module):
        def _hook(module, inp, output):
            # output is the result of c_attn (GPT-2) or v_proj (LLaMA/Qwen)
            B, N, d_out = output.shape
            with torch.no_grad():
                if arch == "gpt2":
                    num_heads = getattr(model.config, "n_head", 12)
                    d_head = d_out // (3 * num_heads)  # c_attn outputs QKV
                    _, _, v = output.split(d_out // 3, dim=2)
                    v_heads = v.view(B, N, num_heads, d_head)
                    V_captured[l_idx] = v_heads.detach().cpu()
                else:  # llama/qwen
                    num_heads = getattr(model.config, "num_attention_heads", 32)
                    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
                    d_head = getattr(model.config, "head_dim", d_out // num_kv_heads)
                    v_heads = output.view(B, N, num_kv_heads, d_head)
                    groups = num_heads // num_kv_heads
                    if groups > 1:
                        v_heads = v_heads.repeat_interleave(groups, dim=2)
                    V_captured[l_idx] = v_heads.detach().cpu()
        return _hook

    layers = list(iter_attn_layers(model, arch))
    for l_idx, (_, attn_module) in enumerate(layers):
        if arch == "gpt2":
            target_mod = attn_module.c_attn
        else:
            target_mod = attn_module.v_proj
        handles.append(target_mod.register_forward_hook(make_hook(l_idx, attn_module)))

    input_ids = input_ids.to(device)
    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    return V_captured, out.attentions, out.logits, out.hidden_states


# ---------------------------------------------------------------------------
# Tier classification — now uses BOTH L∞ and KL
# ---------------------------------------------------------------------------

def classify_tier(htype,
                  attn_linf_nat, output_linf_nat, kl_nat,
                  attn_linf_copy, output_linf_copy, kl_copy):
    """
    Tier 1: safe closed-form substitution on all prompt types.
    Tier 2: safe on natural text, but regime-switches on copy-trigger.
    Tier 3: full attention required.

    Thresholds are differentiated by head type:
      sink  — check attention-weight L∞ + scaled KL proxy
      local — check absolute output L∞ (attn weights are non-uniform) + scaled KL proxy
    """
    if htype == "sink":
        nat_safe  = (attn_linf_nat  < TIER1_SINK_ATTN_LINF and kl_nat  < TIER1_SINK_KL_MAX)
        copy_safe = (attn_linf_copy < TIER1_SINK_ATTN_LINF and kl_copy < TIER1_SINK_KL_MAX)
    elif htype == "local":
        nat_safe  = (output_linf_nat  < TIER1_LOCAL_OUT_LINF and kl_nat  < TIER1_LOCAL_KL_MAX)
        copy_safe = (output_linf_copy < TIER1_LOCAL_OUT_LINF and kl_copy < TIER1_LOCAL_KL_MAX)
    else:  # sink_and_local / unknown — conservative fallback
        nat_safe  = (attn_linf_nat  < TIER1_ATTN_LINF  and
                     output_linf_nat < TIER1_OUTPUT_LINF and kl_nat  < TIER1_KL_MAX)
        copy_safe = (attn_linf_copy < TIER1_ATTN_LINF  and
                     output_linf_copy < TIER1_OUTPUT_LINF and kl_copy < TIER1_KL_MAX)

    if nat_safe and copy_safe:
        return 1
    elif nat_safe and not copy_safe:
        return 2   # regime-switch candidate
    else:
        return 3


# (audit_one_head is no longer used individually, integrated into batched loop)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_natural_prompts(tokenizer, seq_len, num_docs):
    """WikiText-103 held-out TEST split chunked into seq_len-token windows."""
    from datasets import load_dataset
    print(f"  Loading WikiText-103 test split ({num_docs} docs, seq_len={seq_len})...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    full_text = " ".join(row["text"] for row in ds if row["text"].strip())
    all_ids = tokenizer(full_text, return_tensors="pt",
                        add_special_tokens=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i: i + seq_len].unsqueeze(0))
        if len(chunks) >= num_docs:
            break
    print(f"  {len(chunks)} natural chunks")
    return chunks


def build_copy_trigger_prompts(tokenizer, seq_len, num_prompts):
    """
    Copy-trigger prompts: repeated n-gram sequences at various repeat distances.
    Template: [ngram] [filler of `dist` tokens] [ngram - last token]
    Vary: repeat_distance ∈ [5, 20, 100, 500], ngram_len ∈ [1, 2, 3, 4, 5]
    """
    import random
    # Single-token English words (avoids BPE multi-token surprises)
    vocab = [
        "cat", "dog", "mat", "sat", "hat", "bat", "rat", "fat",
        "ran", "can", "pan", "man", "van", "tan", "fan", "ban",
        "tree", "free", "see", "bee", "tea", "sea", "pea", "key",
        "book", "cook", "look", "hook", "took", "good", "wood", "food",
    ]
    repeat_distances = [5, 20, 100, 500]
    ngram_lengths    = [1, 2, 3, 4, 5]
    rng = random.Random(42)
    prompts = []

    for _ in range(num_prompts):
        dist  = rng.choice(repeat_distances)
        n     = rng.choice(ngram_lengths)
        ngram = [rng.choice(vocab) for _ in range(n)]

        filler_words = [rng.choice(vocab) for _ in range(dist)]
        # Query: first ngram + filler + same ngram minus its last token
        # → the model should predict the last ngram token
        query_suffix = ngram[:-1] if len(ngram) > 1 else []
        parts = ngram + filler_words + query_suffix

        text = " ".join(parts)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        # Pad to seq_len with extra random filler so all chunks are the same length
        if ids.shape[1] < seq_len:
            extra = " ".join(rng.choice(vocab) for _ in range(seq_len * 2))
            text = text + " " + extra
            ids  = tokenizer(text, return_tensors="pt",
                             add_special_tokens=True)["input_ids"]
        ids = ids[:, :seq_len]
        prompts.append(ids)

    print(f"  {len(prompts)} copy-trigger prompts")
    return prompts


def build_niah_prompts(tokenizer, seq_len, num_prompts):
    """
    Needle-in-haystack: a key fact buried at positions N/4, N/2, 3N/4.
    Template: [filler] "The secret answer is <X>." [filler] "What is the answer?"
    """
    import random
    rng = random.Random(123)
    filler_sent = "The researchers continued their work on the experiment. "
    insertion_positions = [0.25, 0.5, 0.75]
    prompts = []

    for i in range(num_prompts):
        fact     = rng.randint(10000, 99999)
        needle   = f"The secret answer is {fact}. "
        question = "What is the secret answer? The answer is"

        needle_ids   = tokenizer(needle,   add_special_tokens=False)["input_ids"]
        question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]

        filler_budget = seq_len - len(needle_ids) - len(question_ids) - 5
        if filler_budget < 10:
            filler_budget = 10
        frac    = insertion_positions[i % len(insertion_positions)]
        pre_len = int(filler_budget * frac)
        pst_len = filler_budget - pre_len

        def make_filler(n):
            base = tokenizer(filler_sent * (n // 10 + 2),
                             add_special_tokens=False)["input_ids"]
            return base[:n]

        full_ids = make_filler(pre_len) + needle_ids + make_filler(pst_len) + question_ids
        full_ids = full_ids[:seq_len]
        prompts.append(torch.tensor([full_ids]))

    print(f"  {len(prompts)} NIAH prompts")
    return prompts


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    os.makedirs(PHASE7_DIR, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*68}")
    print(f"  Phase 1 — Per-Head L∞/KL Audit | model={args.model}")
    print(f"  Sink  thresholds : Attn L∞<{TIER1_SINK_ATTN_LINF}, Scaled KL<{TIER1_SINK_KL_MAX}")
    print(f"  Local thresholds : Out L∞<{TIER1_LOCAL_OUT_LINF} (abs), Scaled KL<{TIER1_LOCAL_KL_MAX}")
    print(f"  natural={args.num_natural}, copy={args.num_copy}, niah={args.num_niah}")
    print(f"  device={device}")
    print(f"{'='*68}\n")

    # ---- Load model ----
    model, tokenizer = load_model(args.model, device)
    arch = detect_arch(model)
    print(f"  Detected architecture: {arch}")

    # Resolve actual device (may differ after device_map=auto)
    device = next(model.parameters()).device

    # Count layers/heads
    attn_layers = list(iter_attn_layers(model, arch))
    num_layers  = len(attn_layers)
    # Infer num_heads from first layer
    first_attn = attn_layers[0][1]
    num_heads   = getattr(first_attn, "num_heads",
                  getattr(model.config, "num_attention_heads",
                  getattr(model.config, "n_head", 1)))
    print(f"  Layers: {num_layers}, Heads per layer: {num_heads}")

    # ---- Head type map ----
    head_type_map = {}
    if args.head_type_map and os.path.exists(args.head_type_map):
        with open(args.head_type_map) as f:
            raw = json.load(f)
        for k, v in raw.items():
            layer, head = map(int, k.split(","))
            head_type_map[(layer, head)] = v
        print(f"  Loaded {len(head_type_map)} head type assignments")
    else:
        print("  No head_type_map — auditing every head as both sink AND local")
        for l in range(num_layers):
            for h in range(num_heads):
                head_type_map[(l, h)] = "sink_and_local"

    # ---- Build prompt sets ----
    natural_prompts = build_natural_prompts(tokenizer, args.seq_len, args.num_natural)
    copy_prompts    = build_copy_trigger_prompts(tokenizer, args.seq_len, args.num_copy)
    niah_prompts    = build_niah_prompts(tokenizer, args.seq_len, args.num_niah)

    # ---- Batched Audit Loop ----
    # We invert the loop: for each prompt, we do ONE forward pass to get all Vs and Attns.
    # Then we compute the metrics for all heads. This is ~384x faster.
    
    # metrics structure: metrics[layer_idx][head_idx][htype][dataset] = list of {l_inf, kl}
    from collections import defaultdict
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    datasets = [
        ("natural", natural_prompts),
        ("copy",    copy_prompts),
        ("niah",    niah_prompts)
    ]

    for d_name, prompts in datasets:
        for ids in tqdm(prompts, desc=f"Eval {d_name}"):
            # 1. Single forward pass — captures all Vs, attention weights, and hidden states
            V_dict, attn_tuple, logits, hidden_states = extract_all_V(
                model, ids, arch, device)

            # 2. Precompute local window mask once per prompt (vectorized — no Python loop!)
            N = ids.shape[-1]
            q_idx_t = torch.arange(N, device=device).unsqueeze(1)  # [N, 1]
            k_idx_t = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
            local_mask = (k_idx_t >= q_idx_t - args.local_window + 1) & (k_idx_t <= q_idx_t)
            local_lens = local_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            implied_local_attn = (local_mask.float() / local_lens.float()).unsqueeze(0)  # [1, N, N]

            # 3. Per-layer metrics
            for l_idx in range(num_layers):
                if l_idx not in V_dict or l_idx >= len(attn_tuple):
                    continue
                V_layer    = V_dict[l_idx].to(device)          # [1, N, H, d_head]
                attn_layer = attn_tuple[l_idx].to(device)      # [1, H, N, N]

                # Hidden-state RMS for scaled KL proxy normalization
                hs     = hidden_states[l_idx].to(device)        # [1, N, d_model]
                rms_hs = hs.norm(dim=-1).mean().item() / (hs.shape[-1] ** 0.5)
                rms_hs = max(rms_hs, 1e-8)

                # Retrieve output projection weight helper for this layer
                _, attn_module_l = attn_layers[l_idx]

                for h_idx in range(num_heads):
                    htype_spec = head_type_map.get((l_idx, h_idx), "sink_and_local")
                    htypes = ["sink", "local"] if htype_spec == "sink_and_local" else [htype_spec]

                    # Extract single-head tensors
                    V_head = V_layer[:, :, h_idx, :]         # [1, N, d_head]
                    attn_w = attn_layer[:, h_idx, :, :]      # [1, N, N]
                    d_head = V_head.shape[-1]

                    V4d           = V_head.unsqueeze(1)                       # [1, 1, N, d_head]
                    attn_out_full = torch.bmm(attn_w, V_head).unsqueeze(1)   # [1, 1, N, d_head]

                    for htype in htypes:
                        if htype == "sink":
                            attn_out_sub = sink_substitute(
                                V4d, attn_weights=attn_w.unsqueeze(1),
                                num_sink_tokens=args.num_sink_tokens)

                            # Implied sink: uniform mass on first K tokens
                            implied_attn = torch.zeros_like(attn_w)
                            sink_weights = attn_w[:, :, :args.num_sink_tokens].mean(
                                dim=1, keepdim=True)                          # [1, 1, num_sink]
                            implied_attn[:, :, :args.num_sink_tokens] = sink_weights
                            attn_l_inf = (attn_w - implied_attn).abs().max().item()

                        else:  # local
                            attn_out_sub = local_substitute(V4d, window_size=args.local_window)
                            # Use the precomputed vectorized mask — no per-query Python loop!
                            attn_l_inf = (attn_w - implied_local_attn).abs().max().item()

                        # Absolute output L∞ on per-head value output (pre-o_proj)
                        diff_head    = (attn_out_full - attn_out_sub).squeeze(1)  # [1, N, d_head]
                        output_l_inf = diff_head.abs().max().item()

                        # PROJECTED & SCALED KL proxy:
                        #   Δ_proj = Δ_head × W_o[head]  →  [1, N, d_model]
                        #   kl ≈ ½ (‖Δ_proj‖₂ / rms_hs)²  (Fisher-Info approximation)
                        w_slice = get_head_o_proj_weight(attn_module_l, arch, h_idx, d_head)
                        if w_slice is not None:
                            diff_proj    = torch.matmul(
                                diff_head.float(), w_slice.float())           # [1, N, d_model]
                            delta_scaled = diff_proj.norm(dim=-1).mean().item() / rms_hs
                        else:
                            # Fallback: no projection info available
                            delta_scaled = diff_head.norm(dim=-1).mean().item() / rms_hs
                        kl_approx = 0.5 * delta_scaled ** 2

                        metrics[l_idx][h_idx][htype][d_name].append({
                            "attn_l_inf":   attn_l_inf,
                            "output_l_inf": output_l_inf,
                            "kl_approx":    kl_approx,
                        })

    results      = {}
    summary_rows = []
    all_heads    = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    for (layer_idx, head_idx) in all_heads:
        htype_spec = head_type_map.get((layer_idx, head_idx), "sink_and_local")
        htypes     = ["sink", "local"] if htype_spec == "sink_and_local" else [htype_spec]

        for htype in htypes:
            head_metrics = metrics[layer_idx][head_idx][htype]
            metrics_nat  = head_metrics["natural"]
            metrics_copy = head_metrics["copy"]
            metrics_niah = head_metrics["niah"]

            def agg(lst, key):
                vals = [x[key] for x in lst if x]
                return (np.mean(vals), np.max(vals)) if vals else (float("nan"), float("nan"))

            attn_l_inf_nat_mean,  attn_l_inf_nat_max   = agg(metrics_nat,  "attn_l_inf")
            attn_l_inf_copy_mean, attn_l_inf_copy_max  = agg(metrics_copy, "attn_l_inf")
            attn_l_inf_niah_mean, attn_l_inf_niah_max  = agg(metrics_niah, "attn_l_inf")

            out_l_inf_nat_mean,   out_l_inf_nat_max    = agg(metrics_nat,  "output_l_inf")
            out_l_inf_copy_mean,  out_l_inf_copy_max   = agg(metrics_copy, "output_l_inf")
            out_l_inf_niah_mean,  out_l_inf_niah_max   = agg(metrics_niah, "output_l_inf")

            kl_nat_mean,     kl_nat_max      = agg(metrics_nat,  "kl_approx")
            kl_copy_mean,    kl_copy_max     = agg(metrics_copy, "kl_approx")

            tier = classify_tier(htype,
                                 attn_l_inf_nat_max, out_l_inf_nat_max, kl_nat_max,
                                 attn_l_inf_copy_max, out_l_inf_copy_max, kl_copy_max)

            entry = {
                "layer": layer_idx,
                "head":  head_idx,
                "type":  htype,
                "tier":  tier,
                "attn_l_inf_natural_mean": attn_l_inf_nat_mean,
                "attn_l_inf_natural_max":  attn_l_inf_nat_max,
                "attn_l_inf_copy_max":     attn_l_inf_copy_max,
                "out_l_inf_natural_mean":  out_l_inf_nat_mean,
                "out_l_inf_natural_max":   out_l_inf_nat_max,
                "out_l_inf_copy_max":      out_l_inf_copy_max,
                "kl_natural_mean":    kl_nat_mean,
                "kl_natural_max":     kl_nat_max,
                "kl_copy_mean":       kl_copy_mean,
                "kl_copy_max":        kl_copy_max,
                "n_natural":          len(metrics_nat),
                "n_copy":             len(metrics_copy),
                "n_niah":             len(metrics_niah),
            }
            results[(layer_idx, head_idx, htype)] = entry
            summary_rows.append(entry)

    # ---- Save ----
    json_path = os.path.join(PHASE7_DIR, f"{args.output_prefix}head_audit.json")
    pkl_path  = os.path.join(PHASE7_DIR, f"{args.output_prefix}head_audit.pkl")

    def _nan_safe(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    json_rows = [{k: _nan_safe(v) for k, v in row.items()} for row in summary_rows]
    with open(json_path, "w") as f:
        json.dump(json_rows, f, indent=2)
    with open(pkl_path, "wb") as f:
        pickle.dump({"rows": summary_rows, "by_key": results,
                     "thresholds": {
                         "tier1_sink_attn_linf":  TIER1_SINK_ATTN_LINF,
                         "tier1_sink_kl":         TIER1_SINK_KL_MAX,
                         "tier1_local_out_linf":  TIER1_LOCAL_OUT_LINF,
                         "tier1_local_kl":        TIER1_LOCAL_KL_MAX,
                     }}, f)

    # ---- Summary table ----
    tier_counts = Counter(row["tier"] for row in summary_rows)
    total = len(summary_rows)
    print(f"\n{'='*80}")
    print(f"  HEAD AUDIT SUMMARY — {args.model}")
    print(f"  Sink  Tier-1: Attn L∞<{TIER1_SINK_ATTN_LINF}, Scaled KL<{TIER1_SINK_KL_MAX}  (both nat+copy)")
    print(f"  Local Tier-1: Out L∞<{TIER1_LOCAL_OUT_LINF} (abs), Scaled KL<{TIER1_LOCAL_KL_MAX}  (both nat+copy)")
    print(f"{'='*80}")
    print(f"  Total head×type audited: {total}")
    for t in [1, 2, 3]:
        pct = 100 * tier_counts[t] / max(total, 1)
        label = {1: "safe substitution", 2: "regime-switching", 3: "full attn required"}[t]
        print(f"  Tier {t} ({label:20s}): {tier_counts[t]:4d} / {total}  ({pct:.1f}%)")

    print(f"\n  L∞ / KL table (first 10):")
    print(f"  {'Layer':>5} {'Head':>5} {'Type':>6} {'Tier':>5} "
          f"{'Attn L∞(nat)':>12} {'Out L∞(nat)':>12} {'KL(nat)':>10} {'KL(copy)':>10}")
    print(f"  {'-'*75}")
    for row in summary_rows[:10]:
        def _fmt(v):
            return f"{v:12.5f}" if not math.isnan(v) else "         nan"
        print(f"  {row['layer']:>5} {row['head']:>5} {row['type']:>6} {row['tier']:>5} "
              f"{_fmt(row['attn_l_inf_natural_max'])} {_fmt(row['out_l_inf_natural_max'])} "
              f"{_fmt(row['kl_natural_mean'])} {_fmt(row['kl_copy_mean'])}")

    print(f"\n  Full table: {json_path}")
    print(f"  Pickle:     {pkl_path}")

    # ---- Tier 2 output (for regime_detector.py) ----
    tier2 = [(r["layer"], r["head"], r["type"]) for r in summary_rows if r["tier"] == 2]
    if tier2:
        tier2_path = os.path.join(PHASE7_DIR, f"{args.output_prefix}tier2_heads.json")
        with open(tier2_path, "w") as f:
            json.dump([{"layer": l, "head": h, "type": t} for l, h, t in tier2], f, indent=2)
        print(f"\n  Tier 2 heads ({len(tier2)}) → {tier2_path}")
        print(f"  Run: python phase7/regime_detector.py")

    # ---- Partial summary.json ----
    summary = {
        "model":       args.model,
        "total_heads": total,
        "tier1_count": tier_counts[1],
        "tier2_count": tier_counts[2],
        "tier3_count": tier_counts[3],
        "tier1_pct":   round(100 * tier_counts[1] / max(total, 1), 1),
        "thresholds": {
            "tier1_sink_attn_linf":  TIER1_SINK_ATTN_LINF,
            "tier1_sink_kl":         TIER1_SINK_KL_MAX,
            "tier1_local_out_linf":  TIER1_LOCAL_OUT_LINF,
            "tier1_local_kl":        TIER1_LOCAL_KL_MAX,
        },
    }
    summary_path = os.path.join(PHASE7_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
    else:
        existing = {}
    existing[f"audit_{args.model.replace('/', '_')}"] = summary
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n  Headline numbers appended to: {summary_path}")


if __name__ == "__main__":
    main()
