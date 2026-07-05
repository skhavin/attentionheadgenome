"""
step19_routing_validation.py
------------------------------
Runs the 4-experiment ablation on Qwen2.5-0.5B across 3 benchmarks.

Experiments (one class routed at a time):
  A: Local-stable heads  -> WINDOW_32 (all others: FULL_SOFTMAX)
  B: Sink-stable heads   -> BOS_ROUTE  (all others: FULL_SOFTMAX)
  C: Both Local + Sink   -> cheap      (all others: FULL_SOFTMAX)
  D: All stable heads    -> full routing map from stability.json

Benchmarks:
  1. WikiText-103 perplexity  (in-domain)
  2. HellaSwag accuracy       (OOD commonsense)
  3. ARC-Easy accuracy        (OOD factual QA)

Pre-registered thresholds (from implementation_plan.md, written BEFORE running):
  POSITIVE: HellaSwag delta < 0.5% all models  -> universal claim
  GRAY ZONE: 0.5% - 2.0%                       -> scoped claim
  FAILURE:   > 2.0% on any model               -> domain-specific roles only
"""

import json, os, sys, torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

MODEL      = "Qwen/Qwen2.5-0.5B"
SAFE_MODEL = MODEL.split("/")[-1]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL}...")
tok   = AutoTokenizer.from_pretrained(MODEL)
base_model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
base_model.eval()

# Import routing engine
sys.path.insert(0, "phase2_atlas")
from step18_routing_engine import patch_model, layer_routing, routing_map, WINDOW_SIZE, BOS_BUDGET

import copy

# ── Load stability map ────────────────────────────────────────────────────────
with open(f"outputs/routing/{SAFE_MODEL}_stability.json") as f:
    stab = json.load(f)

def parse_key(k):
    l, h = k.split("_")
    return int(l), int(h)

# Build per-experiment override routing maps
def build_override_map(experiment):
    """
    Returns a per-head routing dict where only the experiment-specific class
    gets cheap attention; all others forced to FULL_SOFTMAX.
    """
    override = {}
    for k, v in stab["routing_map"].items():
        cls = v["class"]
        original_route = v["routing"]
        stab_score = v["stability"]
        
        if experiment == "A":  # Local-only
            route = original_route if (cls == "Local" and stab_score >= 0.85) else "FULL_SOFTMAX"
        elif experiment == "B":  # Sink-only
            route = original_route if (cls == "Sink" and stab_score >= 0.85) else "FULL_SOFTMAX"
        elif experiment == "C":  # Local + Sink
            route = original_route if (cls in ("Local", "Sink") and stab_score >= 0.85) else "FULL_SOFTMAX"
        elif experiment == "D":  # Full routing map
            route = original_route
        else:
            route = "FULL_SOFTMAX"
        override[k] = route
    return override

def make_layer_routing_from_override(override):
    lr = {}
    for k, route in override.items():
        l, h = parse_key(k)
        lr.setdefault(l, {})[h] = route
    return lr

# ── Benchmark functions ───────────────────────────────────────────────────────
def compute_wikitext_ppl(m, max_samples=50):
    with open("outputs/phase2_atlas/dataset.json") as f:
        data = json.load(f)
    texts = [s["text"] for s in data["wikitext"][:max_samples]]
    total_loss, total_tokens = 0.0, 0
    m.eval()
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            if ids["input_ids"].shape[1] < 4:
                continue
            out = m(**ids, labels=ids["input_ids"])
            n_tokens = ids["input_ids"].shape[1]
            total_loss  += out.loss.item() * n_tokens
            total_tokens += n_tokens
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

def compute_hellaswag_acc(m, max_samples=100):
    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(max_samples))
    correct = 0
    m.eval()
    with torch.no_grad():
        for row in ds:
            ctx = row["ctx"]
            endings = row["endings"]
            label = int(row["label"])
            losses = []
            for ending in endings:
                full = ctx + " " + ending
                ids = tok(full, return_tensors="pt", truncation=True, max_length=256).to(device)
                out = m(**ids, labels=ids["input_ids"])
                losses.append(out.loss.item())
            pred = int(np.argmin(losses))
            if pred == label:
                correct += 1
    return correct / max_samples

def compute_arc_acc(m, max_samples=100):
    ds = load_dataset("ai2_arc", "ARC-Easy", split="test").select(range(max_samples))
    correct = 0
    m.eval()
    with torch.no_grad():
        for row in ds:
            question = row["question"]
            choices  = row["choices"]["text"]
            labels_  = row["choices"]["label"]
            answer   = row["answerKey"]
            losses   = []
            for choice in choices:
                full = question + " " + choice
                ids = tok(full, return_tensors="pt", truncation=True, max_length=256).to(device)
                out = m(**ids, labels=ids["input_ids"])
                losses.append(out.loss.item())
            pred_label = labels_[int(np.argmin(losses))]
            if pred_label == answer:
                correct += 1
    return correct / max_samples

# ── Baseline ─────────────────────────────────────────────────────────────────
print("\n=== BASELINE (full softmax everywhere) ===")
base_ppl = compute_wikitext_ppl(base_model)
print(f"WikiText PPL: {base_ppl:.3f}")
base_hs  = compute_hellaswag_acc(base_model)
print(f"HellaSwag:    {base_hs*100:.1f}%")
base_arc = compute_arc_acc(base_model)
print(f"ARC-Easy:     {base_arc*100:.1f}%")

results = {"baseline": {"ppl": base_ppl, "hellaswag": base_hs, "arc": base_arc}}

# ── Experiments ──────────────────────────────────────────────────────────────
from step18_routing_engine import make_routed_forward

for exp in ["A", "B", "C", "D"]:
    print(f"\n=== EXPERIMENT {exp} ===")
    override = build_override_map(exp)
    exp_lr = make_layer_routing_from_override(override)

    # Fresh copy of model
    m = copy.deepcopy(base_model)

    # Patch per-head routing
    num_layers = m.config.num_hidden_layers
    for l in range(num_layers):
        lr = exp_lr.get(l, {})
        if not lr:
            continue
        attn = m.model.layers[l].self_attn
        attn.forward = make_routed_forward(attn.forward, l, m.config, lr)

    n_window   = sum(1 for v in override.values() if v == "WINDOW_32")
    n_bos      = sum(1 for v in override.values() if v == "BOS_ROUTE")
    n_full     = sum(1 for v in override.values() if v == "FULL_SOFTMAX")
    print(f"  WINDOW_32={n_window}, BOS_ROUTE={n_bos}, FULL_SOFTMAX={n_full}")

    ppl = compute_wikitext_ppl(m)
    hs  = compute_hellaswag_acc(m)
    arc = compute_arc_acc(m)

    dppl = ppl  - base_ppl
    dhs  = (hs  - base_hs)  * 100
    darc = (arc - base_arc) * 100

    print(f"  WikiText PPL: {ppl:.3f} (delta={dppl:+.3f})")
    print(f"  HellaSwag:    {hs*100:.1f}% (delta={dhs:+.1f}%)")
    print(f"  ARC-Easy:     {arc*100:.1f}% (delta={darc:+.1f}%)")

    # Pre-registered interpretation
    if abs(dhs) < 0.5 and abs(darc) < 0.5:
        verdict = "POSITIVE: roles generalize fully out-of-domain"
    elif abs(dhs) < 2.0 and abs(darc) < 2.0:
        verdict = "GRAY ZONE: partial generalization, minor degradation"
    else:
        verdict = "FAILURE: domain-specific roles, routing degrades OOD"
    print(f"  Verdict: {verdict}")

    results[f"exp_{exp}"] = {
        "ppl": ppl, "hellaswag": hs, "arc": arc,
        "dppl": dppl, "dhs": dhs, "darc": darc,
        "n_window": n_window, "n_bos": n_bos, "n_full": n_full,
        "verdict": verdict,
    }
    del m

os.makedirs("outputs/routing", exist_ok=True)
with open(f"outputs/routing/{SAFE_MODEL}_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to outputs/routing/{SAFE_MODEL}_validation_results.json")
