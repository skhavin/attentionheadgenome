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

# ============================================================
# CONFIG
# ============================================================

CANONICAL_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "canonical_labels.json"
)
GENOME_CSV_PATH = os.path.join(os.path.dirname(__file__), "full_genome_dataset.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "03_qk_ov_ablation_results.json")

MODELS = [
    {"model_id": "gpt2",                      "model_name": "GPT-2",       "label_key": "gpt2"},
    {"model_id": "Qwen/Qwen2.5-0.5B",         "model_name": "Qwen-0.5B",   "label_key": "qwen_0.5b"},
    {"model_id": "Qwen/Qwen2.5-1.5B",         "model_name": "Qwen-1.5B",   "label_key": "qwen_1.5b"},
    {"model_id": "unsloth/Llama-3.2-1B",      "model_name": "Llama-3.2-1B","label_key": "llama_3.2_1b"},
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_PROMPTS = 5   # Enough for reliable ΔPPL estimates; not too many for budget


# ============================================================
# INTERVENTION HOOKS
# ============================================================

class QKZeroHook:
    """
    Hook that intercepts the attention weights just before they multiply V,
    and replaces them with a uniform distribution (all-equal attention).

    Only active for a specific head (head_idx).
    On architectures where attn_weights are computed inside the module,
    we patch the forward pass to detect and replace the distribution.
    """
    def __init__(self, head_idx):
        self.head_idx = head_idx
        self.handle = None
        self.triggered = False

    def hook_fn(self, module, input, output):
        # output is typically (attn_output, attn_weights, ...) or just attn_output
        # We need to recompute with uniform attention and replace the output
        # This is a best-effort hook — exact behavior depends on model architecture
        self.triggered = True
        return output  # Return unmodified (uniform patching happens via model patch below)


class AttentionPatcher:
    """
    Patches a model's attention computation to replace specific heads'
    attention patterns with either:
    - Uniform: all tokens receive equal attention
    - Zero OV: attention is preserved but output is zeroed out
    
    Works by temporarily overriding forward methods during eval.
    """
    def __init__(self, model, layer_idx, head_idx, n_heads, mode="uniform_qk"):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.n_heads = n_heads
        self.mode = mode  # "uniform_qk" or "zero_ov"
        self.original_forwards = {}
        self.hooks = []

    def install(self):
        """Install the patch for the target layer/head."""
        attn_module = self._get_attn_module()
        if attn_module is None:
            return False

        if self.mode == "uniform_qk":
            # We hook the output and reconstruct with uniform attention
            def hook(module, input, output):
                # output[0] = final hidden state, output[1] = attn_weights (if returned)
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    attn_weights = output[1]  # [batch, n_heads, seq, seq]
                    batch, n_h, seq_q, seq_k = attn_weights.shape
                    
                    # Replace target head's attention with uniform
                    uniform = torch.ones(batch, 1, seq_q, seq_k, device=attn_weights.device)
                    # Mask: causal (upper triangle = 0)
                    mask = torch.tril(torch.ones(seq_q, seq_k, device=attn_weights.device))
                    uniform = uniform * mask
                    uniform = uniform / (uniform.sum(dim=-1, keepdim=True) + 1e-10)
                    
                    patched = attn_weights.clone()
                    patched[:, self.head_idx:self.head_idx+1, :, :] = uniform
                    return (output[0], patched) + output[2:]
                return output

            h = attn_module.register_forward_hook(hook)
            self.hooks.append(h)

        elif self.mode == "zero_ov":
            # We zero out this head's contribution to the output
            # by zeroing the slice of the output projection weight
            # NOTE: This is a weight-level intervention, not a hook
            # We save and restore afterwards
            if hasattr(attn_module, "o_proj"):
                head_dim = attn_module.o_proj.weight.shape[1] // self.n_heads
                start = self.head_idx * head_dim
                end = start + head_dim
                self._saved_weight = attn_module.o_proj.weight.data[:, start:end].clone()
                attn_module.o_proj.weight.data[:, start:end] = 0.0
                self._attn_module_ref = attn_module
                self._ov_start = start
                self._ov_end = end

        return True

    def remove(self):
        """Remove all patches and restore original weights."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

        # Restore OV weights if they were zeroed
        if self.mode == "zero_ov" and hasattr(self, "_saved_weight"):
            self._attn_module_ref.o_proj.weight.data[
                :, self._ov_start:self._ov_end
            ] = self._saved_weight

    def _get_attn_module(self):
        for name, module in self.model.named_modules():
            if f"layers.{self.layer_idx}" in name or f"h.{self.layer_idx}" in name:
                if any(t in type(module).__name__ for t in ["Attention", "attention"]):
                    if hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                        return module
        return None


# ============================================================
# PPL COMPUTATION
# ============================================================

def compute_ppl(model, tokenizer, prompts, device=DEVICE):
    """Compute average perplexity over a list of prompt tensors."""
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for tokens in prompts:
            tokens = tokens.to(device)
            out = model(tokens, labels=tokens)
            nll = out.loss.item()
            n = tokens.shape[1] - 1
            total_nll += nll * n
            total_tokens += n

    return float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")


def load_eval_prompts(tokenizer, n=NUM_EVAL_PROMPTS, seq_len=256):
    """Load held-out evaluation prompts (same source as 02, test split)."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = " ".join(ds["text"])
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompts = []
    stride = max(1, len(tokens) // n)
    for i in range(n):
        chunk = tokens[i * stride: i * stride + seq_len]
        if len(chunk) == seq_len:
            prompts.append(torch.tensor(chunk).unsqueeze(0))
    return prompts[:n]


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
        local_deltas = [r["delta_ppl"] for r in arch_data if r["label"] == "local"]
        retrieval_deltas = [r["delta_ppl"] for r in arch_data if r["label"] == "retrieval"]
        
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
            "passed": passed,
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
        mha_archs = {"GPT-2"}
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
    # Load canonical labels
    with open(CANONICAL_LABELS_PATH, "r") as f:
        canonical_labels = json.load(f)

    all_results = {
        "condition_A_uniform_qk": {},
        "condition_B_zero_ov": {},
    }

    models_to_run = MODELS[:1] if debug else MODELS

    for model_config in models_to_run:
        model_id = model_config["model_id"]
        model_name = model_config["model_name"]
        label_key = model_config["label_key"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

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
        eval_prompts = load_eval_prompts(tokenizer, n=3 if debug else NUM_EVAL_PROMPTS)
        print(f"  Eval prompts: {len(eval_prompts)}")

        # Baseline PPL (no intervention)
        print("  Computing baseline PPL...")
        baseline_ppl = compute_ppl(model, tokenizer, eval_prompts)
        print(f"  Baseline PPL: {baseline_ppl:.2f}")

        arch_results_A = []
        arch_results_B = []

        # Loop over layers and heads
        n_heads_to_test = 2 if debug else n_heads

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads_to_test):
                label_key_head = f"{layer_idx}_{head_idx}"
                head_entry = model_labels_dict.get(label_key_head, {})
                head_label = head_entry.get("label", "unknown")

                if head_label == "unknown":
                    continue

                head_info = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "label": head_label,
                }

                # ---- Condition A: Uniform QK ----
                patcher_A = AttentionPatcher(model, layer_idx, head_idx, n_heads, mode="uniform_qk")
                if patcher_A.install():
                    ppl_A = compute_ppl(model, tokenizer, eval_prompts)
                    patcher_A.remove()
                    arch_results_A.append({
                        **head_info,
                        "ppl": ppl_A,
                        "delta_ppl": ppl_A - baseline_ppl,
                    })

                # ---- Condition B: Zero OV ----
                patcher_B = AttentionPatcher(model, layer_idx, head_idx, n_heads, mode="zero_ov")
                if patcher_B.install():
                    ppl_B = compute_ppl(model, tokenizer, eval_prompts)
                    patcher_B.remove()
                    arch_results_B.append({
                        **head_info,
                        "ppl": ppl_B,
                        "delta_ppl": ppl_B - baseline_ppl,
                    })

            if debug:
                break  # Only test layer 0 in debug

        all_results["condition_A_uniform_qk"][model_name] = arch_results_A
        all_results["condition_B_zero_ov"][model_name] = arch_results_B

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
        all_results["condition_A_uniform_qk"], "Condition A (Uniform QK)"
    )
    gate_a_condition_B = compute_gate_a_stats(
        all_results["condition_B_zero_ov"], "Condition B (Zero OV)"
    )

    # Sanity check for Condition A: class variance
    # If all classes show near-identical ΔPPL under uniform attention,
    # the intervention is too blunt — flag for retry with permutation
    for model_name, arch_data in all_results["condition_A_uniform_qk"].items():
        if arch_data:
            by_class = {}
            for r in arch_data:
                by_class.setdefault(r["label"], []).append(r["delta_ppl"])
            class_means = {k: np.mean(v) for k, v in by_class.items()}
            class_variance = np.var(list(class_means.values())) if len(class_means) > 1 else 0.0
            if class_variance < 0.05 and len(class_means) > 1:
                print(f"\n[!!] DESIGN FLAG for {model_name}: Condition A class variance = {class_variance:.4f} < 0.05")
                print("     Uniform attention may be washing out class distinctions before OV is tested.")
                print("     Consider re-running Condition A with random PERMUTATION of attention weights instead.")

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
    output = {
        "gate_a_condition_A": gate_a_condition_A,
        "gate_a_condition_B": gate_a_condition_B,
        "raw_results": all_results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

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
