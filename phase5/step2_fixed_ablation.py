# -*- coding: utf-8 -*-
# phase5/step2_fixed_ablation.py
#
# FIX: The original ablation hooked the attention OUTPUT (post c_proj), which
# zeroed dead signal because heads were already projected and mixed.
#
# CORRECT APPROACH: Hook c_proj's INPUT (pre-hook) and zero specific head
# slices [h*HEAD_DIM : (h+1)*HEAD_DIM] BEFORE the linear mix. This is true
# head-level causal ablation.
#
# OUTPUTS: outputs/phase5/fixed_ablation.json

import os, sys, json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = r"d:\.cache\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase5")

MODEL_ID   = "gpt2-medium"
NUM_LAYERS = 24
NUM_HEADS  = 16
HEAD_DIM   = 64   # 1024 / 16

THRESHOLD_RETRIEVAL = 0.30
THRESHOLD_INDUCTION = -0.50
THRESHOLD_SINK_ENT  = 0.10

# ── NIAH items ────────────────────────────────────────────────────────────────
NIAH_ITEMS = [
    ("The secret code is ALPHA-7.", "What is the secret code?", "ALPHA-7"),
    ("The magic word is ZELDA.", "What is the magic word?", "ZELDA"),
    ("The password is CRIMSON.", "What is the password?", "CRIMSON"),
    ("The hidden number is 42891.", "What is the hidden number?", "42891"),
    ("The answer is JUPITER.", "What is the answer?", "JUPITER"),
    ("The key phrase is SILVER.", "What is the key phrase?", "SILVER"),
    ("The special token is OMEGA.", "What is the special token?", "OMEGA"),
    ("The activation word is PHOENIX.", "What is the activation word?", "PHOENIX"),
    ("The secret message is FALCON.", "What is the secret message?", "FALCON"),
    ("The magic number is 77531.", "What is the magic number?", "77531"),
]

INDUCTION_PROMPTS = [
    (" A B C A B C A B C A B", "C"),
    (" cat dog bird cat dog bird cat dog bird cat dog", "bird"),
    (" 1 2 3 4 1 2 3 4 1 2 3", "4"),
    (" red blue green red blue green red blue green red blue", "green"),
    (" X Y Z X Y Z X Y Z X", "Y"),
    (" alpha beta gamma alpha beta gamma alpha beta gamma alpha", "beta"),
    (" up down left up down left up down left up", "down"),
    (" square circle triangle square circle triangle square circle", "triangle"),
    (" one two three one two three one two three one", "two"),
    (" Monday Tuesday Wednesday Monday Tuesday Wednesday Monday Tuesday", "Wednesday"),
]

WIKITEXT_PASSAGES = [
    "The history of computing is rich and complex. Early computers filled entire rooms and required teams of engineers to operate. The first electronic computers were developed during World War II for military calculations.",
    "Climate change is one of the most pressing challenges facing humanity. Rising temperatures and extreme weather events are becoming more frequent. Scientists agree that human activity is the primary driver.",
    "The human brain contains approximately 86 billion neurons. These cells communicate through synapses, forming complex networks that underlie thought, emotion, and behavior.",
    "Space exploration has expanded our understanding of the universe. From the moon landing in 1969 to Mars rovers, humanity continues to push the boundaries of what is known.",
    "The development of language is one of the most remarkable features of human evolution. Humans can communicate abstract concepts and transmit knowledge across generations through speech and writing.",
]


def load_labels():
    mech_path = os.path.join(IN_DIR, "gpt2_mechanistic_labels.json")
    if os.path.exists(mech_path):
        with open(mech_path) as f:
            data = json.load(f)
        labels = {}
        for key, role in data["heads"].items():
            l, h = map(int, key.split("_"))
            labels[(l, h)] = role
        if labels:
            print(f"  Loaded {len(labels)} labels from {os.path.basename(mech_path)}")
            return labels

    entropy_path = os.path.join(IN_DIR, "gpt2_retrieval_entropy.json")
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
    print(f"  Derived {len(labels)} labels from entropy file")
    return labels


class FixedHeadAblationHooks:
    """
    Hook c_proj INPUT (pre-hook) to zero specific head slices.
    The input to c_proj has shape [batch, seq, n_heads * head_dim].
    Each head h occupies columns [h*HEAD_DIM : (h+1)*HEAD_DIM].
    Zeroing this BEFORE the projection removes the head's causal contribution.
    """
    def __init__(self, model, heads_to_ablate):
        self.handles = []
        layer_heads = {}
        for l, h in heads_to_ablate:
            layer_heads.setdefault(l, []).append(h)

        for layer_idx, heads in layer_heads.items():
            block = model.transformer.h[layer_idx]
            handle = block.attn.c_proj.register_forward_pre_hook(
                self._make_pre_hook(heads)
            )
            self.handles.append(handle)

    def _make_pre_hook(self, heads):
        def hook(module, input):
            x = input[0].clone()  # [batch, seq, n_heads * head_dim]
            for h in heads:
                start = h * HEAD_DIM
                end   = start + HEAD_DIM
                x[:, :, start:end] = 0.0
            return (x,)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def compute_ppl(model, tokenizer, texts, device):
    nlls = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt").to(device)
        if tokens["input_ids"].shape[1] < 4:
            continue
        with torch.no_grad():
            out = model(**tokens, labels=tokens["input_ids"])
            nlls.append(out.loss.item())
    return float(np.exp(np.mean(nlls))) if nlls else float("nan")


def niah_score(model, tokenizer, device, verbose=False):
    correct = 0
    distractor = "The weather is pleasant today. The birds are singing outside."
    for ctx, question, answer in NIAH_ITEMS:
        text    = f"{ctx} {distractor} {question} Answer:"
        tokens  = tokenizer(text, return_tensors="pt").to(device)
        ans_tok = tokenizer(" " + answer.split("-")[0], add_special_tokens=False)["input_ids"][0]
        with torch.no_grad():
            out  = model(**tokens)
            top5 = out.logits[0, -1, :].topk(5).indices.tolist()
        hit = int(ans_tok in top5)
        correct += hit
        if verbose:
            print(f"  Q: {question:<40} | Ans: {answer:<12} | Top5 hit: {hit}")
    return correct / len(NIAH_ITEMS)


def induction_score(model, tokenizer, device, verbose=False):
    correct = 0
    for prompt, expected in INDUCTION_PROMPTS:
        tokens  = tokenizer(prompt, return_tensors="pt").to(device)
        exp_tok = tokenizer(" " + expected, add_special_tokens=False)["input_ids"][0]
        with torch.no_grad():
            out  = model(**tokens)
            pred = out.logits[0, -1, :].argmax().item()
        hit = int(pred == exp_tok)
        correct += hit
        if verbose:
            print(f"  Prompt: ...{prompt[-25:]:<25} | Exp: {expected:<10} | Hit: {hit}")
    return correct / len(INDUCTION_PROMPTS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID}...")

    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model = model.to(device).eval()

    labels  = load_labels()
    by_role = {"sink": [], "local": [], "retrieval": [], "induction": []}
    for (l, h), role in labels.items():
        by_role[role].append((l, h))

    print("\nHead counts:")
    for role, heads in by_role.items():
        print(f"  {role:<12}: {len(heads)}")

    results = {}

    # ── TEST 1: Retrieval ablation -> NIAH ───────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Retrieval Ablation -> NIAH (FIXED HOOK: c_proj pre-hook)")
    print("="*60)
    print("  Baseline NIAH:")
    baseline_niah = niah_score(model, tok, device, verbose=True)
    print(f"  Baseline NIAH accuracy: {baseline_niah:.4f}")

    hooks = FixedHeadAblationHooks(model, by_role["retrieval"])
    ablated_niah = niah_score(model, tok, device, verbose=True)
    hooks.remove()
    delta_niah = ablated_niah - baseline_niah
    print(f"  Ablated  NIAH accuracy: {ablated_niah:.4f}  (delta={delta_niah:+.4f})")
    results["retrieval_ablation_niah"] = {
        "baseline": round(baseline_niah, 4),
        "ablated":  round(ablated_niah, 4),
        "delta":    round(delta_niah, 4),
        "n_heads_ablated": len(by_role["retrieval"]),
        "hook_type": "c_proj_pre_hook",
    }

    # ── TEST 2: Induction ablation -> Prefix completion ──────────────────────
    print("\n" + "="*60)
    print("TEST 2: Induction Ablation -> Prefix Completion (FIXED HOOK)")
    print("="*60)
    print("  Baseline induction:")
    baseline_ind = induction_score(model, tok, device, verbose=True)
    print(f"  Baseline induction accuracy: {baseline_ind:.4f}")

    hooks = FixedHeadAblationHooks(model, by_role["induction"])
    ablated_ind = induction_score(model, tok, device, verbose=True)
    hooks.remove()
    delta_ind = ablated_ind - baseline_ind
    print(f"  Ablated  induction accuracy: {ablated_ind:.4f}  (delta={delta_ind:+.4f})")
    results["induction_ablation_prefix"] = {
        "baseline": round(baseline_ind, 4),
        "ablated":  round(ablated_ind, 4),
        "delta":    round(delta_ind, 4),
        "n_heads_ablated": len(by_role["induction"]),
        "hook_type": "c_proj_pre_hook",
    }

    # ── TEST 3: Local ablation -> PPL (control, should still explode) ─────────
    print("\n" + "="*60)
    print("TEST 3: Local Ablation -> WikiText PPL (control)")
    print("="*60)
    baseline_ppl = compute_ppl(model, tok, WIKITEXT_PASSAGES, device)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    hooks = FixedHeadAblationHooks(model, by_role["local"])
    ablated_ppl_local = compute_ppl(model, tok, WIKITEXT_PASSAGES, device)
    hooks.remove()
    print(f"  Ablated PPL:  {ablated_ppl_local:.2f}  (delta={ablated_ppl_local - baseline_ppl:+.2f})")
    results["local_ablation_ppl"] = {
        "baseline": round(baseline_ppl, 2),
        "ablated":  round(ablated_ppl_local, 2),
        "delta":    round(ablated_ppl_local - baseline_ppl, 2),
        "n_heads_ablated": len(by_role["local"]),
        "hook_type": "c_proj_pre_hook",
    }

    # ── TEST 4: Sink ablation -> PPL (control, should stay flat) ─────────────
    print("\n" + "="*60)
    print("TEST 4: Sink Ablation -> WikiText PPL (control)")
    print("="*60)
    hooks = FixedHeadAblationHooks(model, by_role["sink"])
    ablated_ppl_sink = compute_ppl(model, tok, WIKITEXT_PASSAGES, device)
    hooks.remove()
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print(f"  Ablated PPL:  {ablated_ppl_sink:.2f}  (delta={ablated_ppl_sink - baseline_ppl:+.2f})")
    results["sink_ablation_ppl"] = {
        "baseline": round(baseline_ppl, 2),
        "ablated":  round(ablated_ppl_sink, 2),
        "delta":    round(ablated_ppl_sink - baseline_ppl, 2),
        "n_heads_ablated": len(by_role["sink"]),
        "hook_type": "c_proj_pre_hook",
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FIXED CAUSAL ABLATION SUMMARY (c_proj pre-hook)")
    print("="*70)
    print(f"  {'Role':<14}  {'Test':<22}  {'Baseline':>10}  {'Ablated':>10}  {'Delta':>10}")
    print(f"  {'-'*68}")
    r = results["retrieval_ablation_niah"]
    print(f"  {'Retrieval':<14}  {'NIAH Accuracy':<22}  {r['baseline']:>10.4f}  {r['ablated']:>10.4f}  {r['delta']:>+10.4f}")
    r = results["induction_ablation_prefix"]
    print(f"  {'Induction':<14}  {'Prefix Completion':<22}  {r['baseline']:>10.4f}  {r['ablated']:>10.4f}  {r['delta']:>+10.4f}")
    r = results["local_ablation_ppl"]
    print(f"  {'Local':<14}  {'WikiText PPL':<22}  {r['baseline']:>10.2f}  {r['ablated']:>10.2f}  {r['delta']:>+10.2f}")
    r = results["sink_ablation_ppl"]
    print(f"  {'Sink':<14}  {'WikiText PPL':<22}  {r['baseline']:>10.2f}  {r['ablated']:>10.2f}  {r['delta']:>+10.2f}")

    out = {
        "model": MODEL_ID,
        "hook_type": "c_proj_pre_hook (fixed)",
        "head_counts": {role: len(h) for role, h in by_role.items()},
        "results": results,
    }
    out_path = os.path.join(OUT_DIR, "fixed_ablation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
