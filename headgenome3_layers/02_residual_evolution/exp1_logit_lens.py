"""
Experiment 1: Sudden vs. Gradual Emergence (Logit Lens)
========================================================
Category B — HeadGenome III

Metric: Per-layer Logit Lens probability assigned to the correct target token.
  P^(l) = softmax(LN(x^(l)) @ W_U)[target]

Emergence layer L*: first layer where target reaches top-1 rank (binary, unambiguous).

Sudden-emergence score S:
  S = delta_max / delta_total
  where delta_max = largest single layer-to-layer probability jump
        delta_total = P^(final) - P^(0)
  S > 0.40 → sudden (one layer accounts for >40% of total gain)

Null control: shuffled-prompt permutation null.
  Same prompts, tokens randomly permuted within each prompt.
  Same target token tracked. Tests whether sharp jumps are real vs. structural artifact.

Pre-registered pass criteria (ALL four required):
  1. Coverage: L* defined for ≥80% of prompts.
  2. Sudden: mean S > 0.40 on prompts with defined L*.
  3. Wilcoxon: S_real > S_shuffled per-prompt, p < 0.05.
  4. Cross-task: criteria 1-3 hold independently on arithmetic AND NIAH.

Partial pass: arithmetic-only pass → proceed to Category C on arithmetic only.
Hard fail: any of 1-3 fail on arithmetic → STOP, reassess.

Run: python headgenome3_layers/02_residual_evolution/exp1_logit_lens.py
     python headgenome3_layers/02_residual_evolution/exp1_logit_lens.py --model qwen-1.5b
     python headgenome3_layers/02_residual_evolution/exp1_logit_lens.py --model gpt2
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from headgenome2_circuits.utils.model_loader import load_model_and_tokenizer

# ------------------------------------------------------------------ #
# Pre-registered thresholds — DO NOT CHANGE after first run          #
# ------------------------------------------------------------------ #
COVERAGE_THRESHOLD  = 0.80   # ≥80% of prompts must have a defined L*
SUDDEN_THRESHOLD    = 0.40   # single layer must account for ≥40% of total gain
P_VALUE_THRESHOLD   = 0.05
TOP_K_DISPLAY       = 5      # how many tokens to track in the trajectory

OUTPUT_DIR         = "outputs/phase3_logit_lens"
ARITH_DATASET_PATH = "headgenome2_circuits/datasets/arithmetic.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# NIAH dataset builder (reused from Experiment 0)                    #
# ------------------------------------------------------------------ #
def build_niah_dataset(n: int = 50, seed: int = 42) -> list:
    random.seed(seed)
    words = ["the", "cat", "sat", "on", "mat", "a", "in", "by", "of", "and"]
    data = []
    for _ in range(n):
        uuid = f"{random.randint(10000, 99999)}-{random.randint(10000, 99999)}"
        pre  = " ".join(random.choices(words, k=20))
        post = " ".join(random.choices(words, k=20))
        prompt = f"{pre} The target code is {uuid}. {post} What is the target code? It is"
        # Target = first digit token of uuid
        data.append({"prompt": prompt, "target_text": uuid.split("-")[0][0]})
    return data

def shuffle_tokens(tokenizer, prompt: str, seed: int) -> str:
    """Return a version of the prompt with tokens randomly permuted."""
    rng = random.Random(seed)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    rng.shuffle(ids)
    return tokenizer.decode(ids)

# ------------------------------------------------------------------ #
# Core computation                                                    #
# ------------------------------------------------------------------ #
def get_target_token_id(tokenizer, target_text: str) -> int:
    """Get the single token id for a short answer string."""
    ids = tokenizer.encode(" " + target_text.strip(), add_special_tokens=False)
    return ids[0] if ids else tokenizer.encode(target_text, add_special_tokens=False)[0]

def run_logit_lens(model, tokenizer, prompt: str, target_id: int,
                   device) -> dict:
    """
    Forward pass with output_hidden_states=True. For each hidden state x^(l),
    apply the final LayerNorm and unembedding to get a probability vector, then
    extract the probability of target_id and its rank.

    Returns:
        {
          "probs":  [float] of length num_layers+1 (embedding + N transformer layers)
          "ranks":  [int]   of length num_layers+1
        }
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden_states = out.hidden_states  # tuple: (embedding, layer_1, ..., layer_N)

    # Final LayerNorm and unembedding
    ln_f  = model.model.norm        # works for Qwen/Llama; GPT2 uses model.transformer.ln_f
    lm_head = model.lm_head         # (vocab, hidden) — no bias

    probs = []
    ranks = []

    for hs in hidden_states:
        last_tok = hs[0, -1, :]    # (hidden,)
        normed   = ln_f(last_tok.unsqueeze(0)).squeeze(0)
        logits   = lm_head(normed)  # (vocab,)
        p = torch.softmax(logits.float(), dim=-1)
        target_prob = float(p[target_id].item())
        rank = int((p > p[target_id]).sum().item())  # 0 = top-1
        probs.append(target_prob)
        ranks.append(rank)

    return {"probs": probs, "ranks": ranks}

def emergence_stats(probs: list, ranks: list) -> dict:
    """
    Given per-layer probs and ranks for one prompt, compute:
      L*     : first layer where rank == 0 (top-1), or None
      S      : delta_max / delta_total (suddenness score)
    """
    n = len(probs)

    # Emergence layer
    l_star = None
    for l, r in enumerate(ranks):
        if r == 0:
            l_star = l
            break

    delta_total = probs[-1] - probs[0]
    if delta_total <= 0:
        s_score = 0.0
    else:
        deltas  = [probs[i] - probs[i-1] for i in range(1, n)]
        delta_max = max(deltas) if deltas else 0.0
        s_score = float(delta_max / delta_total)

    return {"l_star": l_star, "s_score": s_score,
            "delta_total": float(delta_total), "probs": probs, "ranks": ranks}

# ------------------------------------------------------------------ #
# Task runner                                                         #
# ------------------------------------------------------------------ #
def run_on_dataset(model, tokenizer, dataset: list, label: str,
                   device, shuffled: bool = False) -> list:
    """Run logit lens on every prompt in dataset. If shuffled=True, permute tokens first."""
    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"LogitLens [{label}]")):
        target_id = get_target_token_id(tokenizer, item["target_text"])
        prompt = item["prompt"]
        if shuffled:
            prompt = shuffle_tokens(tokenizer, prompt, seed=i)
        try:
            raw = run_logit_lens(model, tokenizer, prompt, target_id, device)
            stats = emergence_stats(raw["probs"], raw["ranks"])
        except Exception as e:
            print(f"  WARNING: prompt {i} failed: {e}")
            stats = {"l_star": None, "s_score": 0.0, "delta_total": 0.0,
                     "probs": [], "ranks": []}
        results.append(stats)
    return results

def gpt2_run_logit_lens(model, tokenizer, prompt: str, target_id: int, device) -> dict:
    """GPT-2 specific path since it uses transformer.ln_f not model.norm."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden_states = out.hidden_states
    ln_f   = model.transformer.ln_f
    lm_head = model.lm_head
    probs, ranks = [], []
    for hs in hidden_states:
        last_tok = hs[0, -1, :]
        normed   = ln_f(last_tok.unsqueeze(0)).squeeze(0)
        logits   = lm_head(normed)
        p = torch.softmax(logits.float(), dim=-1)
        target_prob = float(p[target_id].item())
        rank = int((p > p[target_id]).sum().item())
        probs.append(target_prob)
        ranks.append(rank)
    return {"probs": probs, "ranks": ranks}

# ------------------------------------------------------------------ #
# Gate evaluator                                                      #
# ------------------------------------------------------------------ #
def evaluate_gate(results_real: list, results_null: list, task_label: str,
                  n_layers: int) -> dict:
    s_real  = [r["s_score"] for r in results_real if r["delta_total"] > 1e-6]
    s_null  = [r["s_score"] for r in results_null  if r["delta_total"] > 1e-6]

    l_stars = [r["l_star"] for r in results_real if r["l_star"] is not None]
    l_stars_null = [r["l_star"] for r in results_null if r["l_star"] is not None]
    
    # Accuracy check: how many prompts maintain rank 0 at the final layer?
    final_correct = sum(1 for r in results_real if len(r["ranks"]) > 0 and r["ranks"][-1] == 0)
    final_acc = final_correct / len(results_real) if results_real else 0.0

    coverage = len(l_stars) / len(results_real)
    mean_s   = float(np.mean(s_real)) if s_real else 0.0
    mean_s_null = float(np.mean(s_null)) if s_null else 0.0

    # Wilcoxon: s_real > s_null per-prompt (paired on same prompt index)
    wil_p = 1.0
    try:
        min_len = min(len(s_real), len(s_null))
        if min_len > 1:
            _, wil_p = wilcoxon(s_real[:min_len], s_null[:min_len], alternative="greater")
    except Exception:
        wil_p = 1.0

    coverage_pass  = coverage >= COVERAGE_THRESHOLD
    sudden_pass    = mean_s   >= SUDDEN_THRESHOLD
    wilcoxon_pass  = wil_p    < P_VALUE_THRESHOLD
    gate           = coverage_pass and sudden_pass and wilcoxon_pass

    # Average L* (normalized)
    mean_l_star = float(np.mean(l_stars)) if l_stars else None
    mean_l_star_norm = (mean_l_star / n_layers) if mean_l_star is not None else None
    mean_l_star_null = float(np.mean(l_stars_null)) if l_stars_null else None

    print(f"\n  [{task_label}]")
    print(f"    Coverage (L* defined)  : {coverage:.2%}  (threshold >={COVERAGE_THRESHOLD:.0%})  {'PASS' if coverage_pass else 'FAIL'}")
    print(f"    Final Layer Accuracy   : {final_acc:.2%}  (Sanity check)")
    print(f"    Mean S (real)          : {mean_s:.3f}  (threshold >={SUDDEN_THRESHOLD})  {'PASS' if sudden_pass else 'FAIL'}")
    print(f"    Mean S (null/shuffled) : {mean_s_null:.3f}")
    print(f"    Wilcoxon p             : {wil_p:.4f}  (threshold <{P_VALUE_THRESHOLD})  {'PASS' if wilcoxon_pass else 'FAIL'}")
    if mean_l_star is not None:
        print(f"    Mean L* (real)         : {mean_l_star:.1f}/{n_layers}  ({mean_l_star_norm:.1%} of depth)")
    if mean_l_star_null is not None:
        print(f"    Mean L* (null)         : {mean_l_star_null:.1f}/{n_layers}  (Sanity check: should differ from real)")
    print(f"    GATE: {'PASS ✓' if gate else 'FAIL ✗'}")

    return {
        "task":              task_label,
        "n_prompts":         len(results_real),
        "coverage":          float(coverage),
        "final_acc":         float(final_acc),
        "mean_s_real":       mean_s,
        "mean_s_null":       mean_s_null,
        "wilcoxon_p":        float(wil_p),
        "mean_l_star":       mean_l_star,
        "mean_l_star_norm":  mean_l_star_norm,
        "mean_l_star_null":  mean_l_star_null,
        "l_stars":           l_stars,
        "coverage_pass":     bool(coverage_pass),
        "sudden_pass":       bool(sudden_pass),
        "wilcoxon_pass":     bool(wilcoxon_pass),
        "gate":              bool(gate),
    }

# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def run_experiment_1(model_key: str = "qwen-0.5b", n_prompts: int = 50) -> None:
    print(f"\n{'='*60}")
    print(f"Experiment 1: Logit Lens — Sudden vs. Gradual Emergence")
    print(f"Model: {model_key} | N={n_prompts}")
    print(f"Thresholds: coverage>={COVERAGE_THRESHOLD}, S>={SUDDEN_THRESHOLD}, p<{P_VALUE_THRESHOLD}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_key,
                                                output_attentions=False,
                                                output_hidden_states=True)
    device    = model.device
    n_layers  = model.config.num_hidden_layers

    # Monkey-patch logit_lens for GPT-2 which has a different norm location
    is_gpt2 = "gpt2" in model_key
    _logit_lens = gpt2_run_logit_lens if is_gpt2 else run_logit_lens

    def run_on_dataset_local(dataset, label, shuffled=False):
        results = []
        for i, item in enumerate(tqdm(dataset, desc=f"LogitLens [{label}]")):
            target_id = get_target_token_id(tokenizer, item["target_text"])
            prompt = item["prompt"]
            if shuffled:
                prompt = shuffle_tokens(tokenizer, prompt, seed=i)
            try:
                raw   = _logit_lens(model, tokenizer, prompt, target_id, device)
                stats = emergence_stats(raw["probs"], raw["ranks"])
            except Exception as e:
                print(f"  WARNING [{label}] prompt {i}: {e}")
                stats = {"l_star": None, "s_score": 0.0, "delta_total": 0.0,
                         "probs": [], "ranks": []}
            results.append(stats)
        return results

    # Datasets
    with open(ARITH_DATASET_PATH) as f:
        arith_raw = json.load(f)[:n_prompts]
    arith_dataset = [{"prompt": item["prompt"], "target_text": str(item["answer"])}
                     for item in arith_raw if "answer" in item]
    if not arith_dataset:
        # Fallback: build simple arithmetic prompts if dataset format differs
        arith_dataset = [{"prompt": f"What is {a} plus {b}? Answer:", "target_text": str(a+b)}
                         for a, b in [(i%9+1, i%7+1) for i in range(n_prompts)]][:n_prompts]

    niah_dataset = build_niah_dataset(n_prompts)

    # --- Arithmetic ---
    print("\n--- Arithmetic Task ---")
    arith_real    = run_on_dataset_local(arith_dataset,  "Arith/Real")
    arith_null    = run_on_dataset_local(arith_dataset,  "Arith/Null", shuffled=True)

    # --- NIAH ---
    print("\n--- NIAH Task ---")
    niah_real     = run_on_dataset_local(niah_dataset,   "NIAH/Real")
    niah_null     = run_on_dataset_local(niah_dataset,   "NIAH/Null",  shuffled=True)

    # --- Gate evaluation ---
    print(f"\n{'='*60}\nEXPERIMENT 1 GATE RESULTS\n{'='*60}")
    g_arith = evaluate_gate(arith_real, arith_null, "Arithmetic", n_layers)
    g_niah  = evaluate_gate(niah_real,  niah_null,  "NIAH",       n_layers)

    # Final decision
    print(f"\n{'='*60}\nFINAL GATE DECISION\n{'='*60}")
    if g_arith["gate"] and g_niah["gate"]:
        gate_result = "FULL PASS — proceed to Category C on all tasks."
    elif g_arith["gate"] and not g_niah["gate"]:
        gate_result = "PARTIAL PASS — arithmetic only. Proceed to Category C on arithmetic; hold NIAH until replication."
    else:
        gate_result = "FAILED — do NOT proceed to Category C. Reassess hypothesis."

    print(f"  Arithmetic gate : {'PASS' if g_arith['gate'] else 'FAIL'}")
    print(f"  NIAH gate       : {'PASS' if g_niah['gate'] else 'FAIL'}")
    print(f"\n  VERDICT: {gate_result}")

    # Persist
    def to_python(obj):
        if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [to_python(i) for i in obj]
        if isinstance(obj, bool):  return bool(obj)
        if isinstance(obj, (np.bool_,)):  return bool(obj)
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return obj

    output = to_python({
        "model":           model_key,
        "n_layers":        n_layers,
        "n_prompts":       n_prompts,
        "thresholds": {
            "coverage": COVERAGE_THRESHOLD,
            "sudden":   SUDDEN_THRESHOLD,
            "p_value":  P_VALUE_THRESHOLD,
        },
        "arithmetic": {
            "gate": g_arith,
            "prompt_details": arith_real,
        },
        "niah": {
            "gate": g_niah,
            "prompt_details": niah_real,
        },
        "verdict": gate_result,
        "full_pass": g_arith["gate"] and g_niah["gate"],
        "partial_pass": g_arith["gate"] and not g_niah["gate"],
    })

    out_path = os.path.join(OUTPUT_DIR, f"exp1_logit_lens_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-0.5b",
                        choices=["qwen-0.5b", "qwen-1.5b", "llama-1b", "gemma-2b", "gpt2"],
                        help="Model key from MODELS registry")
    parser.add_argument("--n_prompts", type=int, default=50)
    args = parser.parse_args()
    run_experiment_1(args.model, args.n_prompts)
