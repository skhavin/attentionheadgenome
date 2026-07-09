"""
03_qk_vs_ov_ablation.py

Tests the mechanistic split: WHERE a head reads (QK) vs WHAT it writes (OV).

=== DESIGN DECISION NOTE (read this before modifying) ===
Condition A forces fully UNIFORM attention (zeros out pre-softmax QK scores
so all tokens receive equal weight). This is a deliberately large perturbation.

Known risk: does uniform attention wash out class distinctions before OV is
even tested? A Sink head forced to uniform attention and a Local head forced
to uniform attention both now read from all positions equally — if ΔPPL is
similar for both, it might just mean "uniform attention always hurts" rather
than "OV behavior is class-specific."

Sanity-check: we explicitly compute per-class ΔPPL means AND variances.
If Condition A produces near-identical ΔPPL across all classes (Sink, Local,
Retrieval, Induction), this intervention is too blunt and we should rerun
with Condition A-Retry: randomly PERMUTE attention weights rather than zeroing
them. This preserves the sparsity structure while destroying the content-routing
signal. Script will automatically flag this if it detects class variance in
Condition A results is below 0.05.

Condition B zeros out the OV output (equivalent to ablating what the head
writes to the residual stream while keeping its routing pattern intact).
This is a clean, targeted intervention with no such ambiguity.

Usage:
    python 03_qk_vs_ov_ablation.py --debug     # Single prompt, GPT-2 only
    python 03_qk_vs_ov_ablation.py             # Full run, all 4 architectures
"""

import sys
import os
import json
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import importlib.util

spec = importlib.util.spec_from_file_location(
    "triage_and_schema", 
    os.path.join(os.path.dirname(__file__), "01_triage_and_schema.py")
)
triage_and_schema = importlib.util.module_from_spec(spec)
sys.modules["triage_and_schema"] = triage_and_schema
spec.loader.exec_module(triage_and_schema)

from triage_and_schema import assert_no_bin3_leak
from ablation_utils import compute_head_delta_ppl, q_permutation_fn, ov_zero_fn, test_gqa_isolation, load_wikitext_prompts

# ============================================================
# CONFIG
# ============================================================

CANONICAL_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "canonical_labels.json"
)
GENOME_CSV_PATH = os.path.join(os.path.dirname(__file__), "full_genome_dataset.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "03_qk_ov_ablation_results.json")

MODELS = [
    {"model_id": "gpt2-medium",               "model_name": "GPT-2",       "label_key": "gpt2"},
    {"model_id": "Qwen/Qwen2.5-0.5B",         "model_name": "Qwen-0.5B",   "label_key": "qwen_0.5b"},
    {"model_id": "Qwen/Qwen2.5-1.5B",         "model_name": "Qwen-1.5B",   "label_key": "qwen_1.5b"},
    {"model_id": "unsloth/Llama-3.2-1B",      "model_name": "Llama-3.2-1B","label_key": "llama_3.2_1b"},
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_PROMPTS = 5   # Enough for reliable ΔPPL estimates; not too many for budget


# ============================================================
# GATE A STATISTICS
# ============================================================

def compute_gate_a_stats(results_by_arch, condition_name):
    """
    Compute Mann-Whitney U and Cohen's d for class separation in ΔPPL.
    
    For Gate A: we test whether Retrieval heads show significantly LARGER
    ΔPPL than Local heads under the given intervention (meaning they matter more).
    
    Returns per-architecture pass/fail and a coherent-pattern flag.
    """
    arch_results = {}
    
    for arch_name, arch_data in results_by_arch.items():
        local_deltas = [r["delta_ppl"] for r in arch_data if r["canonical_label"] == "local"]
        retrieval_deltas = [r["delta_ppl"] for r in arch_data if r["canonical_label"] == "retrieval"]
        
        if len(local_deltas) < 3 or len(retrieval_deltas) < 3:
            arch_results[arch_name] = {"status": "insufficient_data", "d": 0.0, "p": 1.0}
            continue
        
        # Mann-Whitney U (non-parametric, safe for small rare-class samples)
        u_stat, p_value = stats.mannwhitneyu(
            retrieval_deltas, local_deltas, alternative="two-sided"
        )
        
        # Cohen's d
        mean_diff = np.mean(retrieval_deltas) - np.mean(local_deltas)
        pooled_std = np.sqrt(
            (np.std(retrieval_deltas)**2 + np.std(local_deltas)**2) / 2
        ) + 1e-10
        d = abs(mean_diff / pooled_std)
        
        passed = (p_value < 0.01) and (d > 0.5)
        arch_results[arch_name] = {
            "passed": bool(passed),
            "d": float(d),
            "p": float(p_value),
            "n_local": len(local_deltas),
            "n_retrieval": len(retrieval_deltas),
            "mean_local_delta": float(np.mean(local_deltas)),
            "mean_retrieval_delta": float(np.mean(retrieval_deltas)),
        }
    
    # Coherent pattern check: which architectures pass?
    # GPT-2 is MHA, Llama is GQA-4, Qwens are GQA-2
    # If GPT-2 (MHA) and Llama (GQA-4) both pass but Qwen-0.5B and Qwen-1.5B fail,
    # this splits along attention-type lines — flag it.
    passing_archs = [a for a, r in arch_results.items() if r.get("passed", False)]
    n_passing = len(passing_archs)
    
    coherence_flag = None
    if n_passing >= 2:
        mha_archs = {"GPT-2-Medium"}
        gqa_archs = {"Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"}
        passing_mha = mha_archs.intersection(passing_archs)
        passing_gqa = gqa_archs.intersection(passing_archs)
        
        # If passes split cleanly along MHA/GQA lines
        if len(passing_mha) > 0 and len(passing_gqa) == 0:
            coherence_flag = "WARN: Only MHA (GPT-2) passes. May reflect MHA-specific artifact, not universal signal. Check against Phase 0 per-arch results before Gate B."
        elif "Qwen-0.5B" in passing_archs and "Qwen-1.5B" not in passing_archs:
            coherence_flag = "WARN: Qwen-0.5B passes but Qwen-1.5B fails despite being architecturally similar. Inconsistent — investigate before Gate B."
        elif "Qwen-1.5B" in passing_archs and "Qwen-0.5B" not in passing_archs:
            coherence_flag = "WARN: Qwen-1.5B passes but Qwen-0.5B fails despite being architecturally similar. Inconsistent — investigate before Gate B."
    
    gate_a_pass = n_passing >= 2
    
    return {
        "condition": condition_name,
        "gate_a_pass": gate_a_pass,
        "n_passing_archs": n_passing,
        "passing_archs": passing_archs,
        "coherence_flag": coherence_flag,
        "per_arch": arch_results,
    }


# ============================================================
# MAIN ABLATION LOOP
# ============================================================

def run_ablation(debug=False):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Load canonical labels
    with open(CANONICAL_LABELS_PATH, "r") as f:
        canonical_labels = json.load(f)
    output = {}
    if os.path.exists(OUTPUT_JSON):
        print(f"Found existing results at {OUTPUT_JSON}. Resuming...")
        with open(OUTPUT_JSON, "r") as f:
            output = json.load(f)

    models_to_run = MODELS[:1] if debug else MODELS

    for model_config in models_to_run:
        model_id = model_config["model_id"]
        model_name = model_config["model_name"]
        label_key = model_config["label_key"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        if model_name in output.get("condition_A_q_permute", {}) and model_name in output.get("condition_B_zero_ov", {}):
            print(f"  Skipping {model_name}, already processed.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        model_data = canonical_labels.get("models", {}).get(model_name, {})
        model_labels_dict = model_data.get("heads", {})

        # Build eval prompts
        print("Loading WikiText control prompts...")
        eval_prompts = load_wikitext_prompts(tokenizer, n=3 if debug else NUM_EVAL_PROMPTS)
        print(f"  Eval prompts: {len(eval_prompts)}")

        # Pre-compute baseline to avoid repeating it
        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for p in eval_prompts:
                p = p.to(DEVICE)
                out = model(p, labels=p)
                nll = out.loss.item()
                n = p.shape[1] - 1
                total_nll += nll * n
                total_tokens += n
        baseline_ppl = float(np.exp(total_nll / total_tokens))
        print(f"  Baseline PPL: {baseline_ppl:.2f}")

        arch_results_A = []
        arch_results_B = []

        # Find our 3-head sanity check targets (only on first model)
        # If debug is true, we ONLY run these 3 targets
        targets = []
        if debug:
            retrieval_head = next((k for k, v in model_labels_dict.items() if v.get("label") == "retrieval"), None)
            local_head = next((k for k, v in model_labels_dict.items() if v.get("label") == "local"), None)
            sink_head = next((k for k, v in model_labels_dict.items() if v.get("label") == "sink"), None)
            if retrieval_head: targets.append(retrieval_head)
            if local_head: targets.append(local_head)
            if sink_head: targets.append(sink_head)
            print(f"  [DRY RUN] Running exactly 3 heads: {targets}")
            
            # GQA Test
            test_gqa_isolation(model, tokenizer)

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                label_key_head = f"{layer_idx}_{head_idx}"
                head_entry = model_labels_dict.get(label_key_head, {})
                head_label = head_entry.get("label", "unknown")

                if head_label == "unknown":
                    continue
                    
                if debug and label_key_head not in targets:
                    continue

                # Condition A: Q Permutation
                res_A = compute_head_delta_ppl(
                    model, tokenizer, eval_prompts, layer_idx, head_idx,
                    intervention_fn=q_permutation_fn, target="q",
                    baseline_ppl=baseline_ppl, architecture=model_name,
                    prompt_id="wikitext-test", label=head_label, dry_run=debug,
                    condition_name="q_permutation"
                )
                arch_results_A.append(res_A)

                # Condition B: Zero OV
                res_B = compute_head_delta_ppl(
                    model, tokenizer, eval_prompts, layer_idx, head_idx,
                    intervention_fn=ov_zero_fn, target="ov",
                    baseline_ppl=baseline_ppl, architecture=model_name,
                    prompt_id="wikitext-test", label=head_label, dry_run=debug,
                    condition_name="ov_zero"
                )
                arch_results_B.append(res_B)

            if debug and len(arch_results_A) >= len(targets):
                break

        if "condition_A_q_permute" not in output: output["condition_A_q_permute"] = {}
        if "condition_B_zero_ov" not in output: output["condition_B_zero_ov"] = {}
        output["condition_A_q_permute"][model_name] = arch_results_A
        output["condition_B_zero_ov"][model_name] = arch_results_B

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ============================================================
    # GATE A STATISTICAL ANALYSIS
    # ============================================================
    print("\n" + "="*60)
    print("GATE A ANALYSIS")
    print("="*60)

    gate_a_condition_A = compute_gate_a_stats(
        output["condition_A_q_permute"], "Condition A (Q Permute)"
    )
    gate_a_condition_B = compute_gate_a_stats(
        output["condition_B_zero_ov"], "Condition B (Zero OV)"
    )

    # Sanity check for Condition A: class variance
    for model_name, arch_data in output["condition_A_q_permute"].items():
        if arch_data:
            by_class = {}
            for r in arch_data:
                by_class.setdefault(r["canonical_label"], []).append(r["delta_ppl"])
            class_means = {k: np.mean(v) for k, v in by_class.items()}
            class_variance = np.var(list(class_means.values())) if len(class_means) > 1 else 0.0
            if class_variance < 0.05 and len(class_means) > 1:
                print(f"\n[!!] DESIGN FLAG for {model_name}: Condition A class variance = {class_variance:.4f} < 0.05")

    # Print Gate A summary
    for gate_result in [gate_a_condition_A, gate_a_condition_B]:
        print(f"\n--- {gate_result['condition']} ---")
        print(f"  Gate A PASS: {gate_result['gate_a_pass']} "
              f"({gate_result['n_passing_archs']}/4 architectures)")
        print(f"  Passing archs: {gate_result['passing_archs']}")
        if gate_result["coherence_flag"]:
            print(f"  >> {gate_result['coherence_flag']}")
        for arch_name, arch_result in gate_result["per_arch"].items():
            if "d" in arch_result:
                status = "PASS" if arch_result.get("passed") else "FAIL"
                print(f"    {arch_name}: {status} | d={arch_result['d']:.3f} | p={arch_result['p']:.4f} "
                      f"| n_retrieval={arch_result.get('n_retrieval', 0)}")

    # Save all results
    output_final = {
        "gate_a_condition_A": gate_a_condition_A,
        "gate_a_condition_B": gate_a_condition_B,
        "raw_results": output,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_final, f, indent=2)

    print(f"\n[DONE] Results saved to: {OUTPUT_JSON}")

    gate_A_pass = gate_a_condition_A["gate_a_pass"]
    gate_B_pass = gate_a_condition_B["gate_a_pass"]
    print(f"\n{'='*60}")
    print(f"GATE A VERDICT: {'PROCEED TO GATE B' if (gate_A_pass or gate_B_pass) else 'DO NOT PROCEED — no signal found'}")
    print(f"  Condition A (Uniform QK): {'PASS' if gate_A_pass else 'FAIL'}")
    print(f"  Condition B (Zero OV):    {'PASS' if gate_B_pass else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_ablation(debug=args.debug)
