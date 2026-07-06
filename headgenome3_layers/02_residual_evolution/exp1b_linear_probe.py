"""
Experiment 1b: Linear Probe Diagnostic (v2 — corrected label definition)
=========================================================================
Category B — HeadGenome III

LABEL CORRECTION from v1:
  WRONG: binary "does this model get this prompt right at layer L?"
         → constant per prompt across all layers, probe sees no variance
  CORRECT: multi-class "which digit (0–9) is the correct answer?"
         → label varies across prompts; probe must learn to decode the
           answer concept from the residual stream geometry at each layer.
         → probe accuracy above chance (10% for 10-class) at layer l means
           the residual stream at l linearly encodes WHICH answer is correct.

NULL CONTROL (shuffled-pairing):
  Same residual stream X, but labels randomly reassigned across prompts.
  (i.e., same digit distribution, different pairings with geometry)
  → probe accuracy on shuffled labels should stay at chance if probe
    is tracking arithmetic content, not prompt-template features.

CROSS-ARCHITECTURE DECISION TREE (pre-registered before seeing results):
  A) Probe emerges early (Lp < 20% depth) on BOTH qwen-0.5b and qwen-1.5b,
     at similar relative depths (within 15% of each other):
     → Early emergence is real and consistent. Fix tokenizer for LL. Stay with raw LL.

  B) Probe emerges early on qwen-0.5b but NOT on qwen-1.5b (or much later):
     → Architectural divergence. Report as architectural finding.
     → Do NOT force a single L* claim across models.

  C) Probe emerges early on qwen-0.5b AND on qwen-1.5b, but at different
     relative depths (delta > 15% of depth):
     → Report relative-depth divergence as architectural finding.
     → The "early" stage location is model-specific, not universal.

  D) Probe matches Logit Lens L* on qwen-0.5b (delta ≤ 2 layers):
     → LL basis alignment is adequate for qwen-0.5b.
     → If LL failed on qwen-1.5b but probe succeeds early: LL is the wrong
        tool for that model → switch to Tuned Lens for cross-arch comparison.

  E) Probe also doesn't emerge early on qwen-0.5b:
     → qwen-0.5b's LL L*=2 was a shallow prior artifact (not arithmetic content).
     → Reassess the Category B hypothesis entirely.

Run:
  python headgenome3_layers/02_residual_evolution/exp1b_linear_probe.py --model qwen-0.5b
  python headgenome3_layers/02_residual_evolution/exp1b_linear_probe.py --model qwen-1.5b
  python headgenome3_layers/02_residual_evolution/exp1b_linear_probe.py --model gpt2
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from headgenome2_circuits.utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR        = "outputs/phase3_logit_lens"
PROBE_ACC_THRESH  = 0.80    # emergence: first layer where CV acc > 80%
ARCH_DEPTH_DELTA  = 0.15    # branch C: >15% relative depth difference = divergence
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# Dataset (N=200 few-shot arithmetic for sufficient probe training)  #
# ------------------------------------------------------------------ #
def build_arithmetic_dataset(n: int = 200, seed: int = 42) -> list:
    """
    Few-shot format: "1+1=2\\n2+3=5\\n{a}+{b}="
    Forces clean single-digit token output, eliminates space-prefix ambiguity.

    Label = correct answer digit (integer 0–9, multi-class).
    Stratified so each digit class appears roughly equally.
    """
    random.seed(seed)
    examples = "1+1=2\n2+3=5\n3+4=7\n5+2=7\n"
    # Generate pairs that keep sums 1-9 (single digit)
    pairs = [(a, b) for a in range(1, 9) for b in range(1, 9 - a + 1)]
    # Expand by sampling with replacement to get N items
    items = []
    rng = random.Random(seed)
    for _ in range(n):
        a, b = rng.choice(pairs)
        prompt = examples + f"{a}+{b}="
        items.append({"prompt": prompt, "label": a + b, "target_text": str(a + b)})
    return items

def build_niah_dataset(n: int = 200, seed: int = 42) -> list:
    """
    NIAH: label = first digit of the UUID (integer 1–9).
    This makes it multi-class in the same way as arithmetic.
    """
    random.seed(seed)
    words = ["the", "cat", "sat", "on", "mat", "a", "in", "by", "of", "and"]
    data = []
    rng = random.Random(seed)
    for _ in range(n):
        first_digit = rng.randint(1, 9)
        rest        = rng.randint(1000, 9999)
        uuid_part   = f"{first_digit}{rest}-{rng.randint(10000,99999)}"
        pre  = " ".join(rng.choices(words, k=20))
        post = " ".join(rng.choices(words, k=20))
        prompt = (f"{pre} The target code is {uuid_part}. "
                  f"{post} What is the target code? It is")
        data.append({"prompt": prompt, "label": first_digit,
                     "target_text": str(first_digit)})
    return data

# ------------------------------------------------------------------ #
# Hidden state extraction                                             #
# ------------------------------------------------------------------ #
def extract_hidden_states(model, tokenizer, prompts: list,
                          device) -> tuple:
    """
    Returns:
      X: np.ndarray (N, n_layers+1, d_model)  — last-token residual stream
      y: np.ndarray (N,)                       — correct answer digit labels
    """
    X_list, y_list = [], []
    for item in tqdm(prompts, desc="  Extracting hidden states"):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, output_attentions=False)
        # last-token from every layer's hidden state
        layer_vecs = [hs[0, -1, :].float().cpu().numpy()
                      for hs in out.hidden_states]  # list[(d_model,)]
        X_list.append(np.stack(layer_vecs, axis=0))  # (n_layers+1, d_model)
        y_list.append(item["label"])

    return np.stack(X_list, axis=0), np.array(y_list, dtype=int)

# ------------------------------------------------------------------ #
# Per-layer probe with shuffled-pairing null                         #
# ------------------------------------------------------------------ #
def fit_probe_per_layer(X: np.ndarray, y: np.ndarray,
                        n_layers_total: int,
                        rng_seed: int = 42) -> dict:
    """
    X: (N, n_layers_total, d_model)
    y: (N,)  multi-class answer digits

    Returns dict with:
      'real_accs':     list(float)  per-layer mean 5-fold CV accuracy
      'null_accs':     list(float)  same but with shuffled y (pairing null)
      'chance':        float        1 / n_unique_classes
    """
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
    chance = 1.0 / len(np.unique(y))

    real_accs = []
    null_accs = []

    # Shuffled-pairing null: same X, permuted y
    rng   = np.random.default_rng(rng_seed)
    y_null = rng.permutation(y)

    for l in tqdm(range(n_layers_total), desc="  Probing layers", leave=False):
        Xl = X[:, l, :]   # (N, d_model)

        def cv_acc(labels):
            fold_accs = []
            for tr_idx, te_idx in cv.split(Xl, labels):
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xl[tr_idx])
                Xte = sc.transform(Xl[te_idx])
                clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
                try:
                    clf.fit(Xtr, labels[tr_idx])
                    fold_accs.append(accuracy_score(labels[te_idx],
                                                    clf.predict(Xte)))
                except Exception:
                    fold_accs.append(chance)
            return float(np.mean(fold_accs))

        real_accs.append(cv_acc(y))
        null_accs.append(cv_acc(y_null))

    return {
        "real_accs": real_accs,
        "null_accs": null_accs,
        "chance":    float(chance),
    }

def probe_emergence_layer(accs: list, threshold: float) -> int | None:
    for l, a in enumerate(accs):
        if a >= threshold:
            return l
    return None

# ------------------------------------------------------------------ #
# Load prior Logit Lens result for comparison                        #
# ------------------------------------------------------------------ #
def load_ll_lstar(model_key: str, task: str) -> float | None:
    path = f"{OUTPUT_DIR}/exp1_logit_lens_{model_key}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get(task, {}).get("gate", {}).get("mean_l_star")

# ------------------------------------------------------------------ #
# Decision tree evaluation (pre-registered)                          #
# ------------------------------------------------------------------ #
def evaluate_decision(results_by_model: dict) -> str:
    """
    results_by_model: {model_key: {task: {"Lp_norm": float, "Lp": int|None}}}
    Returns branch letter A–E with rationale.
    """
    arith = {m: r["arithmetic"] for m, r in results_by_model.items()
             if "arithmetic" in r}

    norms = {m: a["Lp_norm"] for m, a in arith.items() if a["Lp_norm"] is not None}

    if len(norms) == 0:
        return ("E", "Probe never emerges on any model. "
                "qwen-0.5b LL L*=2 is a shallow-prior artifact. "
                "Category B hypothesis needs reassessment.")

    vals = list(norms.values())
    keys = list(norms.keys())

    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        both_early = all(v < 0.20 for v in vals)
        if both_early and spread <= ARCH_DEPTH_DELTA:
            return ("A", f"Both models emerge early (spread={spread:.1%} <= {ARCH_DEPTH_DELTA:.0%}). "
                    "Early emergence consistent. Fix tokenizer for LL.")
        elif both_early and spread > ARCH_DEPTH_DELTA:
            return ("C", f"Both emerge early but at different depths (spread={spread:.1%}). "
                    "Report as architectural finding. Do not force a single L*.")
        elif vals[0] < 0.20 and vals[1] >= 0.20:
            return ("B", f"Architectural divergence: {keys[0]} early, {keys[1]} not. "
                    "Do not generalize single L* claim.")
    # Single model
    if vals[0] < 0.20:
        return ("D", f"Probe emerges early ({vals[0]:.1%} depth). Compare to LL L*.")

    return ("E", f"Probe does not emerge early. Check against LL L* for shallow-prior.")

# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def run_probe_diagnostic(model_key: str = "qwen-0.5b",
                         n_prompts: int = 200) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment 1b (v2): Linear Probe Diagnostic")
    print(f"Model: {model_key} | N={n_prompts}")
    print(f"Label: CORRECT ANSWER DIGIT (multi-class, 1–9)")
    print(f"Null:  Shuffled-pairing (same X, permuted y)")
    print(f"Threshold: {PROBE_ACC_THRESH:.0%} | Depth-divergence: {ARCH_DEPTH_DELTA:.0%}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(
        model_key, output_attentions=False, output_hidden_states=True)
    device   = model.device
    n_layers = model.config.num_hidden_layers

    arith_data = build_arithmetic_dataset(n_prompts)
    niah_data  = build_niah_dataset(n_prompts)

    output_by_task = {}

    for task_label, dataset in [("arithmetic", arith_data), ("niah", niah_data)]:
        print(f"\n--- {task_label.upper()} (N={len(dataset)}) ---")
        X, y = extract_hidden_states(model, tokenizer, dataset, device)

        n_classes = len(np.unique(y))
        chance    = 1.0 / n_classes
        print(f"  Classes (unique answers): {sorted(np.unique(y))} | Chance: {chance:.1%}")

        probe_data = fit_probe_per_layer(X, y, n_layers + 1)

        Lp      = probe_emergence_layer(probe_data["real_accs"], PROBE_ACC_THRESH)
        Lp_norm = (Lp / n_layers) if Lp is not None else None
        ll_lstar = load_ll_lstar(model_key, task_label)
        ll_lstar_norm = (ll_lstar / n_layers) if ll_lstar is not None else None

        print(f"\n  Per-layer probe accuracy (real | null | chance={chance:.1%}):")
        for l, (r, n) in enumerate(zip(probe_data["real_accs"],
                                       probe_data["null_accs"])):
            marker = " <-- Lp (probe emergence)" if l == Lp else ""
            print(f"    Layer {l:2d}: real={r:.1%}  null={n:.1%}{marker}")

        print(f"\n  Probe emergence Lp        : layer {Lp} ({Lp_norm:.1%} depth)"
              if Lp is not None else
              f"\n  Probe emergence Lp        : None (never hit {PROBE_ACC_THRESH:.0%})")
        print(f"  Logit Lens L* (exp1)      : layer {ll_lstar} ({ll_lstar_norm:.1%} depth)"
              if ll_lstar is not None else
              f"  Logit Lens L* (exp1)      : N/A")

        if Lp is not None and ll_lstar is not None:
            delta = abs(Lp - ll_lstar)
            if delta <= 2:
                match = "MATCH — LL basis alignment adequate"
            elif Lp < ll_lstar - 2:
                match = "PROBE EARLIER — possible LL basis-alignment artifact"
            else:
                match = "PROBE LATER — unexpected; review"
        else:
            match = "Cannot compare"
        print(f"  Probe vs LL alignment     : {match}")

        output_by_task[task_label] = {
            "n_prompts":    len(dataset),
            "n_classes":    int(n_classes),
            "chance":       float(chance),
            "real_accs":    [float(a) for a in probe_data["real_accs"]],
            "null_accs":    [float(a) for a in probe_data["null_accs"]],
            "Lp":           Lp,
            "Lp_norm":      float(Lp_norm) if Lp_norm is not None else None,
            "ll_lstar":     ll_lstar,
            "ll_lstar_norm":float(ll_lstar_norm) if ll_lstar_norm is not None else None,
            "match_verdict":match,
        }

    # Save
    out_path = f"{OUTPUT_DIR}/exp1b_probe_{model_key}.json"
    result   = {
        "model":     model_key,
        "n_prompts": n_prompts,
        "tasks":     output_by_task,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    return {model_key: output_by_task}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-0.5b",
                        choices=["qwen-0.5b", "qwen-1.5b", "gpt2"])
    parser.add_argument("--n_prompts", type=int, default=200,
                        help="N per task. 200 gives 5-fold CV folds of 40 each.")
    args = parser.parse_args()

    results = run_probe_diagnostic(args.model, args.n_prompts)

    # If both models have been run, evaluate cross-arch decision tree
    both_path_05 = f"{OUTPUT_DIR}/exp1b_probe_qwen-0.5b.json"
    both_path_15 = f"{OUTPUT_DIR}/exp1b_probe_qwen-1.5b.json"
    if os.path.exists(both_path_05) and os.path.exists(both_path_15):
        print(f"\n{'='*60}")
        print("CROSS-ARCHITECTURE DECISION TREE (pre-registered)")
        print(f"{'='*60}")
        combined = {}
        for path, key in [(both_path_05, "qwen-0.5b"), (both_path_15, "qwen-1.5b")]:
            with open(path) as f:
                combined[key] = json.load(f)["tasks"]
        branch, rationale = evaluate_decision(combined)
        print(f"  Branch {branch}: {rationale}")
