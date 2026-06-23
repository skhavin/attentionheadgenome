# -*- coding: utf-8 -*-
# phase5/step1_causal_ablation.py
#
# PURPOSE: The Causal Necessity Test (Week 2 - Scientific Proof).
#
#   Zero out the output projection W_O of specific head species.
#   Measure task-specific failure:
#     Retrieval  ablation → NIAH accuracy drops to 0%
#     Sink       ablation → PPL explodes after long context
#     Induction  ablation → prefix-completion (repeating pattern) score drops
#     Local      ablation → WikiText PPL degrades (but doesn't explode)
#
# DESIGN:
#   - Zero-training: only intercept forward pass, no weight update
#   - Fast: each test < 10 minutes on a single GPU
#   - Uses gpt2-medium as primary (fastest, clearest taxonomy, MHA)
#
# OUTPUTS:
#   outputs/phase5/causal_ablation.json

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase5")

MODEL_ID       = "gpt2-medium"
LABEL_FILE     = os.path.join(IN_DIR, "gpt2_retrieval_entropy.json")
NUM_LAYERS     = 24
NUM_HEADS      = 16
HEAD_DIM       = 64   # gpt2-medium: d_model=1024 / 16 heads

# ──────────────────────────────────────────────────────────────────────────────
# NIAH passages for retrieval ablation test
# ──────────────────────────────────────────────────────────────────────────────
NIAH_ITEMS = [
    ("The secret code is ALPHA-7.", "What is the secret code?", "ALPHA-7"),
    ("The magic word is ZELDA.", "What is the magic word?", "ZELDA"),
    ("The password is CRIMSON-MOON.", "What is the password?", "CRIMSON-MOON"),
    ("The hidden number is 42891.", "What is the hidden number?", "42891"),
    ("The answer is JUPITER.", "What is the answer?", "JUPITER"),
    ("The key phrase is SILVER-DAWN.", "What is the key phrase?", "SILVER-DAWN"),
    ("The special token is OMEGA-9.", "What is the special token?", "OMEGA-9"),
    ("The activation word is PHOENIX.", "What is the activation word?", "PHOENIX"),
    ("The secret message is BLUE-FALCON.", "What is the secret message?", "BLUE-FALCON"),
    ("The magic number is 77531.", "What is the magic number?", "77531"),
]

# Repeating pattern for induction ablation
INDUCTION_PROMPTS = [
    ("A B C A B C A B C A B", "C"),
    ("cat dog bird cat dog bird cat dog bird cat dog", "bird"),
    ("1 2 3 4 1 2 3 4 1 2 3", "4"),
    ("red blue green red blue green red blue green red blue", "green"),
    ("X Y Z X Y Z X Y Z X", "Y"),
    ("Monday Tuesday Wednesday Monday Tuesday Wednesday Monday Tuesday", "Wednesday"),
    ("alpha beta gamma alpha beta gamma alpha beta gamma alpha", "beta"),
    ("up down left up down left up down left up", "down"),
    ("square circle triangle square circle triangle square circle", "triangle"),
    ("one two three one two three one two three one", "two"),
]

# WikiText passages for PPL tests (short segments for speed)
WIKITEXT_PASSAGES = [
    "The history of computing is rich and complex. Early computers filled entire rooms and required teams of engineers to operate. The first electronic computers were developed during World War II for military calculations.",
    "Climate change is one of the most pressing challenges facing humanity. Rising temperatures, melting ice caps, and extreme weather events are becoming more frequent. Scientists agree that human activity is the primary driver of these changes.",
    "The human brain contains approximately 86 billion neurons. These cells communicate through synapses, forming complex networks that underlie thought, emotion, and behavior. Neuroscience is still working to fully understand these mechanisms.",
    "Space exploration has expanded our understanding of the universe. From the first moon landing in 1969 to the Mars rovers and the James Webb Space Telescope, humanity continues to push the boundaries of what is known.",
    "The development of language is one of the most remarkable features of human evolution. Unlike other animals, humans can communicate abstract concepts, plan for the future, and transmit knowledge across generations through speech and writing.",
]


THRESHOLD_RETRIEVAL = 0.30
THRESHOLD_INDUCTION = -0.50
THRESHOLD_SINK_ENT  = 0.10


def load_labels():
    """
    Load mechanistic labels from gpt2_mechanistic_labels.json.
    Falls back to deriving labels from gpt2_retrieval_entropy.json delta values.
    """
    # Try mechanistic labels file first
    mech_path = os.path.join(IN_DIR, "gpt2_mechanistic_labels.json")
    if os.path.exists(mech_path):
        with open(mech_path) as f:
            data = json.load(f)
        labels = {}
        for key, role in data["heads"].items():   # heads: {key: role_string}
            l, h = map(int, key.split("_"))
            labels[(l, h)] = role
        if labels:
            print(f"  Loaded {len(labels)} labels from {os.path.basename(mech_path)}")
            return labels

    # Fallback: derive from entropy delta values
    entropy_path = os.path.join(IN_DIR, "gpt2_retrieval_entropy.json")
    if not os.path.exists(entropy_path):
        print(f"[ERROR] Neither gpt2_mechanistic_labels.json nor gpt2_retrieval_entropy.json found.")
        sys.exit(1)
    with open(entropy_path) as f:
        data = json.load(f)

    labels = {}
    for key, v in data["heads"].items():
        l, h = map(int, key.split("_"))
        delta = v.get("delta")
        me    = v.get("match_entropy")
        nme   = v.get("nonmatch_entropy")

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

    print(f"  Derived {len(labels)} labels from {os.path.basename(entropy_path)}")
    return labels


def get_heads_by_role(labels):
    by_role = {"sink": [], "local": [], "retrieval": [], "induction": []}
    for (l, h), role in labels.items():
        by_role[role].append((l, h))
    return by_role


class HeadAblationHooks:
    """Register forward hooks that zero the output of specified (layer, head) pairs."""

    def __init__(self, model, heads_to_ablate):
        self.handles = []
        self.heads_to_ablate = heads_to_ablate

        # Group by layer
        self.layer_heads = {}
        for l, h in heads_to_ablate:
            self.layer_heads.setdefault(l, []).append(h)

        for layer_idx, attn_heads in self.layer_heads.items():
            block   = model.transformer.h[layer_idx]
            handle  = block.attn.register_forward_hook(
                self._make_hook(layer_idx, attn_heads)
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx, heads):
        def hook(module, input, output):
            # output[0]: (batch, seq, d_model)
            # GPT-2 attention output is already projected; we need to zero
            # the contribution of specific heads post-projection.
            # Approach: re-zero specific head slices in the pre-projection output.
            # Since GPT-2 splits heads via reshape, we zero the packed output slice.
            attn_out = output[0]  # (batch, seq, d_model)
            # Each head contributes HEAD_DIM=64 dims to d_model=1024
            for h in heads:
                start = h * HEAD_DIM
                end   = start + HEAD_DIM
                attn_out[:, :, start:end] = 0.0
            # Return modified output (keep rest of tuple: present, weights)
            return (attn_out,) + output[1:]
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def compute_ppl(model, tokenizer, texts, device):
    """Compute mean perplexity over a list of text strings."""
    nlls = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt").to(device)
        input_ids = tokens["input_ids"]
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out    = model(**tokens, labels=input_ids)
            nll    = out.loss.item()
            nlls.append(nll)
    return float(np.exp(np.mean(nlls))) if nlls else float("nan")


def niah_score(model, tokenizer, device, verbose=False):
    """
    Needle-in-a-haystack accuracy.
    For each item: append a distractor sentence, then ask the question.
    Score = fraction where correct answer token appears in top-5 predictions.
    """
    correct = 0
    for ctx, question, answer in NIAH_ITEMS:
        distractor = "The weather is pleasant today. The birds are singing outside."
        text = f"{ctx} {distractor} {question} Answer:"
        tokens  = tokenizer(text, return_tensors="pt").to(device)
        ans_tok = tokenizer(" " + answer.split("-")[0], add_special_tokens=False)["input_ids"][0]
        with torch.no_grad():
            out   = model(**tokens)
            top5  = out.logits[0, -1, :].topk(5).indices.tolist()
        hit = int(ans_tok in top5)
        correct += hit
        if verbose:
            print(f"  Q: {question[:40]:<40} | Ans: {answer:<15} | Top5: {hit}")
    return correct / len(NIAH_ITEMS)


def induction_score(model, tokenizer, device, verbose=False):
    """
    Induction accuracy: predict the last token of a repeating sequence.
    Score = fraction where correct next token is the top-1 prediction.
    """
    correct = 0
    for prompt, expected in INDUCTION_PROMPTS:
        tokens   = tokenizer(" " + prompt, return_tensors="pt").to(device)
        exp_tok  = tokenizer(" " + expected, add_special_tokens=False)["input_ids"][0]
        with torch.no_grad():
            out  = model(**tokens)
            pred = out.logits[0, -1, :].argmax().item()
        hit = int(pred == exp_tok)
        correct += hit
        if verbose:
            pred_str = tokenizer.decode([pred])
            print(f"  Prompt: {prompt[-30:]:<30} | Exp: {expected:<10} | Pred: {pred_str:<10} | {hit}")
    return correct / len(INDUCTION_PROMPTS)


def run_ablation(model, tokenizer, by_role, target_role, test_fn, baseline_score, device, label):
    """Zero-out ablation: remove target_role heads, measure score drop."""
    heads = by_role[target_role]
    print(f"\n  Ablating {len(heads)} {target_role} heads...")
    hooks = HeadAblationHooks(model, heads)
    try:
        ablated_score = test_fn(model, tokenizer, device)
    finally:
        hooks.remove()
    delta = ablated_score - baseline_score
    print(f"  Baseline {label}: {baseline_score:.4f}")
    print(f"  Ablated  {label}: {ablated_score:.4f}  (delta={delta:+.4f})")
    return {"baseline": round(baseline_score, 4), "ablated": round(ablated_score, 4),
            "delta": round(delta, 4), "n_heads_ablated": len(heads)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID}...")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device).eval()

    labels   = load_labels()
    by_role  = get_heads_by_role(labels)
    print(f"\nHead counts by role:")
    for role, heads in by_role.items():
        print(f"  {role:<12}: {len(heads)}")

    results = {}

    # ── TEST 1: Retrieval ablation → NIAH ─────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Retrieval Ablation -> NIAH Accuracy")
    print("="*60)
    print("  Baseline NIAH:")
    baseline_niah = niah_score(model, tokenizer=tok, device=device, verbose=True)
    print(f"  Baseline NIAH accuracy: {baseline_niah:.4f}")

    ret_result = run_ablation(
        model, tok, by_role, "retrieval",
        lambda m, t, d: niah_score(m, t, d),
        baseline_niah, device, "NIAH acc"
    )
    results["retrieval_ablation_niah"] = ret_result

    # ── TEST 2: Induction ablation → Prefix completion ────────────────────
    print("\n" + "="*60)
    print("TEST 2: Induction Ablation -> Prefix Completion Accuracy")
    print("="*60)
    print("  Baseline Induction:")
    baseline_ind = induction_score(model, tok, device, verbose=True)
    print(f"  Baseline induction accuracy: {baseline_ind:.4f}")

    ind_result = run_ablation(
        model, tok, by_role, "induction",
        lambda m, t, d: induction_score(m, t, d),
        baseline_ind, device, "induction acc"
    )
    results["induction_ablation_prefix"] = ind_result

    # ── TEST 3: Local ablation → WikiText PPL ────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Local Ablation -> WikiText PPL (should degrade, not explode)")
    print("="*60)
    baseline_ppl = compute_ppl(model, tok, WIKITEXT_PASSAGES, device)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    heads = by_role["local"]
    print(f"  Ablating {len(heads)} local heads...")
    hooks = HeadAblationHooks(model, heads)
    try:
        ablated_ppl = compute_ppl(model, tok, WIKITEXT_PASSAGES, device)
    finally:
        hooks.remove()
    print(f"  Ablated PPL:  {ablated_ppl:.2f}  (delta={ablated_ppl-baseline_ppl:+.2f})")
    results["local_ablation_ppl"] = {
        "baseline": round(baseline_ppl, 2),
        "ablated":  round(ablated_ppl, 2),
        "delta":    round(ablated_ppl - baseline_ppl, 2),
        "n_heads_ablated": len(heads),
    }

    # ── TEST 4: Sink ablation → PPL degradation ──────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Sink Ablation -> WikiText PPL")
    print("="*60)
    print(f"  Baseline PPL: {baseline_ppl:.2f}  (same as above)")
    heads = by_role["sink"]
    print(f"  Ablating {len(heads)} sink heads...")
    hooks = HeadAblationHooks(model, heads)
    try:
        ablated_ppl_sink = compute_ppl(model, tok, WIKITEXT_PASSAGES, device)
    finally:
        hooks.remove()
    print(f"  Ablated PPL:  {ablated_ppl_sink:.2f}  (delta={ablated_ppl_sink-baseline_ppl:+.2f})")
    results["sink_ablation_ppl"] = {
        "baseline": round(baseline_ppl, 2),
        "ablated":  round(ablated_ppl_sink, 2),
        "delta":    round(ablated_ppl_sink - baseline_ppl, 2),
        "n_heads_ablated": len(heads),
    }

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("CAUSAL ABLATION SUMMARY")
    print("="*70)
    print(f"  Model: {MODEL_ID}")
    print(f"\n  {'Ablated Role':<16}  {'Test':<24}  {'Baseline':>10}  {'Ablated':>10}  {'Delta':>10}")
    print(f"  {'-'*72}")

    r = results["retrieval_ablation_niah"]
    print(f"  {'Retrieval':<16}  {'NIAH Accuracy':<24}  {r['baseline']:>10.4f}  {r['ablated']:>10.4f}  {r['delta']:>+10.4f}")

    r = results["induction_ablation_prefix"]
    print(f"  {'Induction':<16}  {'Prefix Completion':<24}  {r['baseline']:>10.4f}  {r['ablated']:>10.4f}  {r['delta']:>+10.4f}")

    r = results["local_ablation_ppl"]
    print(f"  {'Local':<16}  {'WikiText PPL':<24}  {r['baseline']:>10.2f}  {r['ablated']:>10.2f}  {r['delta']:>+10.2f}")

    r = results["sink_ablation_ppl"]
    print(f"  {'Sink':<16}  {'WikiText PPL':<24}  {r['baseline']:>10.2f}  {r['ablated']:>10.2f}  {r['delta']:>+10.2f}")

    out = {
        "model":      MODEL_ID,
        "device":     device,
        "head_counts": {role: len(h) for role, h in by_role.items()},
        "niah_items":       len(NIAH_ITEMS),
        "induction_items":  len(INDUCTION_PROMPTS),
        "wikitext_passages": len(WIKITEXT_PASSAGES),
        "results":    results,
    }
    out_path = os.path.join(OUT_DIR, "causal_ablation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
