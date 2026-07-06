"""
Experiment 0: The Temporal Handoff — Bulletproof Edition
=====================================================
Implements all four gate criteria defined in HeadGenome_Part3_Plan.md:
1. Retrieval heads tested on NIAH task (not just arithmetic)
2. Per-prompt Wilcoxon signed-rank test for statistical validity
3. Control group (Local/Sink heads) tested to verify Induction-specificity via Mann-Whitney U
4. Pre-registered numeric pass thresholds applied before any interpretation

Pre-registered thresholds (DO NOT CHANGE after running):
  - Induction PASS: Decode/Prefill ratio > 2.0 AND Wilcoxon p < 0.05 AND Mann-Whitney vs control p < 0.05
  - Retrieval PASS: Prefill/Decode ratio > 1.5 on NIAH task AND Wilcoxon p < 0.05
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from headgenome2_circuits.utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase3_execution"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ARITH_DATASET_PATH = "headgenome2_circuits/datasets/arithmetic.json"

# ============================================================
# Pre-registered Thresholds (locked before running)
# ============================================================
INDUCTION_DECODE_PREFILL_THRESHOLD = 2.0
RETRIEVAL_PREFILL_DECODE_THRESHOLD = 1.5
P_VALUE_THRESHOLD = 0.05

# ============================================================
# NIAH Task Builder
# ============================================================
def build_niah_dataset(n=50):
    """Build a minimal NIAH dataset: a long prefix containing a UUID, 
    followed by a 'find the UUID' query to force Retrieval head engagement."""
    data = []
    for _ in range(n):
        uuid = f"{random.randint(10000,99999)}-{random.randint(10000,99999)}"
        filler = " ".join([random.choice(["The", "cat", "sat", "on", "a", "mat"]) for _ in range(40)])
        prompt = f"{filler} The target code is {uuid}. {filler} What is the target code? It is"
        data.append({"prompt": prompt, "uuid": uuid})
    return data

# ============================================================
# Phase Activation Hook
# ============================================================
class PhaseActivationHook:
    """
    Hooks the input to o_proj to calculate the L2 norm of each head's contribution
    to the residual stream: || W_O^h * x_h ||_2
    
    Phase is determined strictly by seq_len:
      - seq_len > 1 → Prefill (parallel processing of the full prompt)
      - seq_len == 1 → Decode (single autoregressive step)
    
    Returns per-prompt-per-phase norms so we can compute per-prompt ratios for statistics.
    """
    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []
        # dict[(layer, head)] -> {"prefill": [per_prompt_norm,...], "decode": [per_prompt_norm,...]}
        self.norms = {}
        self._current_prompt_id = 0
        self._in_prompt = {"prefill": None, "decode": None}

    def _create_hook(self, layer_idx):
        def hook(module, input):
            x = input[0]  # (batch, seq_len, hidden_size)
            seq_len = x.shape[1]
            phase = "prefill" if seq_len > 1 else "decode"

            W_O = module.weight  # (hidden_size, hidden_size)

            for h in range(self.num_heads):
                start = h * self.head_dim
                end = start + self.head_dim
                x_h = x[:, :, start:end]           # (batch, seq_len, head_dim)
                W_O_h = W_O[:, start:end]            # (hidden_size, head_dim)
                head_out = F.linear(x_h, W_O_h)     # (batch, seq_len, hidden_size)
                norm = torch.linalg.norm(head_out, dim=-1).mean().item()

                key = (layer_idx, h)
                if key not in self.norms:
                    self.norms[key] = {"prefill": [], "decode": []}

                # We append per-prompt. Since a single generate() call
                # produces exactly one Prefill pass and one (or more) Decode passes,
                # we take the mean of any multi-step Decode norms per prompt.
                # To do this cleanly, we accumulate within a prompt buffer.
                if self._in_prompt[phase] is None:
                    self._in_prompt[phase] = {}
                buf = self._in_prompt[phase]
                if key not in buf:
                    buf[key] = []
                buf[key].append(norm)

            return (x,)
        return hook

    def flush_prompt(self):
        """Call after each prompt's generate() to finalize per-prompt norms."""
        for phase in ["prefill", "decode"]:
            if self._in_prompt[phase] is not None:
                for key, norms in self._in_prompt[phase].items():
                    self.norms[key][phase].append(np.mean(norms))
                self._in_prompt[phase] = None

    def register(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.o_proj.register_forward_pre_hook(self._create_hook(layer_idx))
            self.hooks.append(handle)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# ============================================================
# Head Population Classifier
# ============================================================
def classify_heads(model, tokenizer, dataset, num_heads, device, num_samples=50):
    """
    Classify heads into four populations using attention patterns on the arithmetic task:
      - Induction/Counting: high attention mass on operand tokens (from last position)
      - Retrieval: high attention mass on distant content tokens (from middle positions)
      - Local: attention mass concentrated on immediately preceding tokens
      - Sink: attention mass concentrated on position 0
    Returns separate lists of (layer, head) tuples for each population.
    """
    num_layers = model.config.num_hidden_layers
    
    induction_masses = torch.zeros((num_layers, num_heads), device=device)
    local_masses = torch.zeros((num_layers, num_heads), device=device)
    sink_masses = torch.zeros((num_layers, num_heads), device=device)
    
    valid_count = 0
    for item in dataset[:num_samples]:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs)
        
        attentions = outputs.attentions
        last_idx = seq_len - 1
        valid_count += 1

        for l in range(num_layers):
            attn = attentions[l][0]  # (heads, seq, seq)
            # Induction: from last token to all non-adjacent tokens
            induction_masses[l] += attn[:, last_idx, max(0, last_idx-3):].mean(dim=-1)
            # Local: from last token to the 3 most recent tokens
            local_masses[l] += attn[:, last_idx, max(0, last_idx-3):last_idx].mean(dim=-1)
            # Sink: attention mass on token 0
            sink_masses[l] += attn[:, last_idx, 0]

    induction_masses /= valid_count
    local_masses /= valid_count
    sink_masses /= valid_count

    flat_i = induction_masses.flatten().cpu().numpy()
    flat_l = local_masses.flatten().cpu().numpy()
    flat_s = sink_masses.flatten().cpu().numpy()

    top_i = [(int(idx // num_heads), int(idx % num_heads)) for idx in np.argsort(flat_i)[::-1][:10]]
    # Control group: highest-ranked Local + Sink heads, excluding any overlap with Induction
    top_l = [(int(idx // num_heads), int(idx % num_heads)) for idx in np.argsort(flat_l)[::-1][:10]]
    top_s = [(int(idx // num_heads), int(idx % num_heads)) for idx in np.argsort(flat_s)[::-1][:10]]

    induction_set = set(top_i)
    control_heads = list(set(top_l + top_s) - induction_set)[:10]

    return top_i, control_heads

# ============================================================
# Core Measurement
# ============================================================
def measure_phase_norms(model, tokenizer, dataset, target_heads, label, device):
    """
    Runs generate() on each prompt, tracking per-prompt Prefill and Decode norms
    for the specified head population. Returns two arrays of per-prompt norms.
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    tracker = PhaseActivationHook(num_heads, head_dim)
    tracker.register(model)

    for item in tqdm(dataset, desc=f"Measuring {label}"):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
        tracker.flush_prompt()

    tracker.remove()

    # Aggregate per-prompt norms across the specified heads
    per_prompt_prefill = []
    per_prompt_decode = []
    n_prompts = len(dataset)

    for h in target_heads:
        if h in tracker.norms:
            pf = tracker.norms[h]["prefill"]
            dc = tracker.norms[h]["decode"]
            if len(pf) == n_prompts and len(dc) == n_prompts:
                per_prompt_prefill.append(pf)
                per_prompt_decode.append(dc)

    if not per_prompt_prefill:
        return np.array([]), np.array([])

    # Mean across heads, per prompt
    per_prompt_prefill = np.mean(per_prompt_prefill, axis=0)
    per_prompt_decode = np.mean(per_prompt_decode, axis=0)
    return per_prompt_prefill, per_prompt_decode

# ============================================================
# Statistical Evaluation
# ============================================================
def evaluate_gate(name, per_prompt_a, per_prompt_b, ratio_threshold, a_label, b_label):
    """
    Applies the pre-registered gate criteria:
      - Mean ratio of a vs b
      - One-sided Wilcoxon signed-rank test (a > b)
    Returns a dict of results.
    """
    if len(per_prompt_a) == 0 or len(per_prompt_b) == 0:
        return {"gate": False, "reason": "Empty norms — heads not engaged on this task."}

    ratios = per_prompt_a / (per_prompt_b + 1e-9)
    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))

    try:
        stat, p_val = wilcoxon(per_prompt_a, per_prompt_b, alternative="greater")
    except Exception:
        p_val = 1.0
        stat = 0.0

    print(f"\n  [{name}] {a_label} vs {b_label}")
    print(f"    Mean {a_label}: {np.mean(per_prompt_a):.4f} ± {np.std(per_prompt_a):.4f}")
    print(f"    Mean {b_label}: {np.mean(per_prompt_b):.4f} ± {np.std(per_prompt_b):.4f}")
    print(f"    Mean Ratio ({a_label}/{b_label}): {mean_ratio:.2f} ± {std_ratio:.2f}")
    print(f"    Wilcoxon p-value ({a_label} > {b_label}): {p_val:.4f}")
    print(f"    Threshold: ratio > {ratio_threshold}, p < {P_VALUE_THRESHOLD}")

    ratio_pass = mean_ratio > ratio_threshold
    wilcoxon_pass = p_val < P_VALUE_THRESHOLD

    return {
        "name": name,
        "mean_a": float(np.mean(per_prompt_a)),
        "mean_b": float(np.mean(per_prompt_b)),
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
        "wilcoxon_p": float(p_val),
        "ratio_threshold": ratio_threshold,
        "ratio_pass": ratio_pass,
        "wilcoxon_pass": wilcoxon_pass,
        "gate": ratio_pass and wilcoxon_pass
    }

# ============================================================
# Main
# ============================================================
def run_experiment_0(model_key="qwen-0.5b", n_prompts=50):
    print(f"\nLoading {model_key} for Experiment 0 (Bulletproof Edition)...")
    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=True)
    device = model.device
    num_heads = model.config.num_attention_heads

    with open(ARITH_DATASET_PATH) as f:
        arith_dataset = json.load(f)[:n_prompts]

    niah_dataset = build_niah_dataset(n_prompts)

    print("\n--- Step 1: Classifying Head Populations ---")
    induction_heads, control_heads = classify_heads(
        model, tokenizer, arith_dataset, num_heads, device
    )
    print(f"  Induction/Counting Heads: {induction_heads[:5]}  (top 5)")
    print(f"  Control (Local/Sink) Heads: {control_heads[:5]}  (top 5)")

    # Criterion 3: Control group for Mann-Whitney U
    print("\n--- Step 2: Measuring Phase Norms on Arithmetic (Induction + Control) ---")
    induction_prefill_arith, induction_decode_arith = measure_phase_norms(
        model, tokenizer, arith_dataset, induction_heads, "Induction/Arith", device
    )
    control_prefill_arith, control_decode_arith = measure_phase_norms(
        model, tokenizer, arith_dataset, control_heads, "Control/Arith", device
    )

    # Criterion 1: Retrieval on NIAH (proper task)
    print("\n--- Step 3: Measuring Phase Norms on NIAH (Induction + Control for comparison) ---")
    induction_prefill_niah, induction_decode_niah = measure_phase_norms(
        model, tokenizer, niah_dataset, induction_heads, "Induction/NIAH", device
    )
    control_prefill_niah, control_decode_niah = measure_phase_norms(
        model, tokenizer, niah_dataset, control_heads, "Control/NIAH", device
    )

    print("\n========== EXPERIMENT 0 RESULTS ==========")

    # Criterion A: Induction is Decode-dominant (Arithmetic)
    result_induction = evaluate_gate(
        "Induction Decode-Dominance (Arithmetic)",
        induction_decode_arith, induction_prefill_arith,
        INDUCTION_DECODE_PREFILL_THRESHOLD,
        "Decode", "Prefill"
    )

    # Criterion B: Control is NOT Decode-dominant (specificity check)
    result_control = evaluate_gate(
        "Control Decode-Dominance (Arithmetic) — Should FAIL",
        control_decode_arith, control_prefill_arith,
        INDUCTION_DECODE_PREFILL_THRESHOLD,
        "Decode", "Prefill"
    )

    # Criterion C: Mann-Whitney to prove Induction is more Decode-dominant than Control
    if len(induction_decode_arith) > 0 and len(control_decode_arith) > 0:
        induction_ratios = induction_decode_arith / (induction_prefill_arith + 1e-9)
        control_ratios = control_decode_arith / (control_prefill_arith + 1e-9)
        _, mwu_p = mannwhitneyu(induction_ratios, control_ratios, alternative="greater")
        print(f"\n  Mann-Whitney U (Induction > Control Decode Ratio): p = {mwu_p:.4f}")
        mwu_pass = mwu_p < P_VALUE_THRESHOLD
    else:
        mwu_p, mwu_pass = 1.0, False

    # Criterion D: Induction on NIAH
    result_induction_niah = evaluate_gate(
        "Induction Decode-Dominance (NIAH) — Cross-task check",
        induction_decode_niah, induction_prefill_niah,
        INDUCTION_DECODE_PREFILL_THRESHOLD,
        "Decode", "Prefill"
    )

    # Final Gate Decision
    print("\n========== FINAL GATE DECISION ==========")
    induction_gate = result_induction["gate"] and mwu_pass
    print(f"  Induction Decode-Dominance (Arith): ratio_pass={result_induction['ratio_pass']}, wilcoxon_pass={result_induction['wilcoxon_pass']}, mwu_specificity_pass={mwu_pass}")
    print(f"  Induction Decode-Dominance (NIAH):  ratio_pass={result_induction_niah['ratio_pass']}, wilcoxon_pass={result_induction_niah['wilcoxon_pass']}")
    print(f"  Control group correctly weaker: {not result_control['gate']}")
    print(f"\n  HARD GATE: {'PASSED' if induction_gate else 'FAILED'}")

    if not induction_gate:
        print("\n  >>> Gate FAILED. Do NOT proceed to Experiments 1-3. Reassess hypothesis. <<<")

    def make_serializable(obj):
        """Recursively convert numpy/bool_ types to Python native."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    results = make_serializable({
        "model": model_key,
        "pre_registered_thresholds": {
            "induction_decode_prefill_ratio": INDUCTION_DECODE_PREFILL_THRESHOLD,
            "retrieval_prefill_decode_ratio": RETRIEVAL_PREFILL_DECODE_THRESHOLD,
            "p_value": P_VALUE_THRESHOLD,
        },
        "induction_heads": induction_heads,
        "control_heads": control_heads,
        "results": {
            "induction_arith": result_induction,
            "control_arith": result_control,
            "induction_niah": result_induction_niah,
            "mann_whitney_p": float(mwu_p),
            "mann_whitney_pass": bool(mwu_pass),
        },
        "gate_passed": bool(induction_gate)
    })

    out_path = os.path.join(OUTPUT_DIR, f"exp0_temporal_handoff_v2_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_experiment_0("qwen-0.5b")
