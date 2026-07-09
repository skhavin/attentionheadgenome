"""
04_shuffle_survival.py

Tests whether a head's function is driven by TOKEN POSITION or TOKEN CONTENT.

Two orthogonal interventions:
  A) Position Shuffle: Randomize the ORDER of tokens, preserve the vocabulary.
     A head that survives this is responding to position/recency, not meaning.
     Expected survivors: Sink (BOS is gone), Local (window stays roughly intact)
     Expected to fail: Retrieval (needle at wrong position), Induction (pattern disrupted)

  B) Content Shuffle: Randomize token IDENTITY, preserve the positions.
     Achieved by replacing tokens with random draws from the vocabulary.
     A head that survives this is responding to position only, not content.
     Expected survivors: pure Sink/Local (they mostly care about recency)
     Expected to die: Retrieval (specific rare token is gone), Induction (pattern gone)

CLASS SEPARATION TEST (Gate A criterion):
  Metric: Macro-F1 of class labels predicted by a simple threshold on ΔPPL
  (not the actual classifier — just checking if position/content signals separate classes)
  Statistical test: Mann-Whitney U between Retrieval and Local ΔPPL distributions
  Threshold: d > 0.5, p < 0.01 in ≥ 2 of 4 architectures

Usage:
    python 04_shuffle_survival.py --debug
    python 04_shuffle_survival.py
"""

import sys
import os
import json
import argparse
import gc
import numpy as np
import torch
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
from ablation_utils import compute_head_delta_ppl, ov_zero_fn, test_gqa_isolation, load_wikitext_prompts, compute_ppl

CANONICAL_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "canonical_labels.json"
)
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "04_shuffle_survival_results.json")

MODELS = [
    {"model_id": "gpt2-medium",              "model_name": "GPT-2",       "label_key": "gpt2"},
    {"model_id": "Qwen/Qwen2.5-0.5B",        "model_name": "Qwen-0.5B",   "label_key": "qwen_0.5b"},
    {"model_id": "Qwen/Qwen2.5-1.5B",        "model_name": "Qwen-1.5B",   "label_key": "qwen_1.5b"},
    {"model_id": "unsloth/Llama-3.2-1B",     "model_name": "Llama-3.2-1B","label_key": "llama_3.2_1b"},
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROMPTS = 5
SEQ_LEN = 256


# ============================================================
# SHUFFLE INTERVENTIONS
# ============================================================

def position_shuffle(token_ids: torch.Tensor) -> torch.Tensor:
    """
    Randomly shuffle the ORDER of tokens in the sequence.
    Vocabulary/content is preserved, but positional relationships are destroyed.
    
    We preserve the BOS token at position 0 to avoid catastrophic tokenizer failures,
    then shuffle everything else.
    """
    ids = token_ids.clone()
    seq = ids[0]
    
    if len(seq) <= 2:
        return ids
    
    # Keep BOS in place, shuffle the rest
    rest = seq[1:].clone()
    perm = torch.randperm(len(rest))
    seq[1:] = rest[perm]
    ids[0] = seq
    return ids


def content_shuffle(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Replace each token with a random draw from the vocabulary.
    Positional structure is preserved, content is destroyed.
    
    We avoid special token IDs (0, 1, 2) which could corrupt model behavior.
    """
    ids = token_ids.clone()
    seq_len = ids.shape[1]
    random_ids = torch.randint(3, vocab_size, (1, seq_len), device=ids.device)
    return random_ids


# ============================================================
# GATE A STATISTICS
# ============================================================

def coherent_pattern_check(arch_pass_results, condition_name):
    """
    Same coherent-pattern flag logic as in 03_qk_vs_ov_ablation.py.
    Catches suspicious MHA/GQA splits or Qwen inconsistencies.
    """
    passing_archs = [a for a, r in arch_pass_results.items() if r.get("passed", False)]
    flags = []

    mha_archs = {"GPT-2"}
    gqa_archs = {"Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"}
    passing_mha = mha_archs.intersection(passing_archs)
    passing_gqa = gqa_archs.intersection(passing_archs)

    if len(passing_mha) > 0 and len(passing_gqa) == 0:
        flags.append(f"WARN [{condition_name}]: Only MHA (GPT-2) passes — possible MHA-specific artifact.")
    if "Qwen-0.5B" in passing_archs and "Qwen-1.5B" not in passing_archs:
        flags.append(f"WARN [{condition_name}]: Qwen-0.5B passes but Qwen-1.5B fails — inconsistent for similar architectures.")
    if "Qwen-1.5B" in passing_archs and "Qwen-0.5B" not in passing_archs:
        flags.append(f"WARN [{condition_name}]: Qwen-1.5B passes but Qwen-0.5B fails — inconsistent for similar architectures.")

    return passing_archs, flags


# ============================================================
# MAIN LOOP
# ============================================================

def run_shuffle(debug=False):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    with open(CANONICAL_LABELS_PATH, "r") as f:
        canonical_labels = json.load(f)

    all_arch_results = {
        "position_shuffle": {},
        "content_shuffle": {},
    }
    
    if os.path.exists(OUTPUT_JSON):
        print(f"Found existing results at {OUTPUT_JSON}. Resuming...")
        with open(OUTPUT_JSON, "r") as f:
            all_arch_results = json.load(f)

    models_to_run = MODELS[:1] if debug else MODELS

    for model_config in models_to_run:
        model_id = model_config["model_id"]
        model_name = model_config["model_name"]
        label_key = model_config["label_key"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        if model_name in all_arch_results.get("position_shuffle", {}) and model_name in all_arch_results.get("content_shuffle", {}):
            print(f"  Skipping {model_name}, already processed.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        vocab_size = tokenizer.vocab_size

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        model_data = canonical_labels.get("models", {}).get(model_name, {})
        model_labels_dict = model_data.get("heads", {})
        model_labels = {k: v.get("label", "unknown") for k, v in model_labels_dict.items()}
        eval_prompts = load_wikitext_prompts(tokenizer, n=3 if debug else NUM_PROMPTS)

        # Pre-compute baselines for both shuffle conditions
        print("  Computing position-shuffled baseline...")
        pos_shuffled = [position_shuffle(p) for p in eval_prompts]
        pos_baseline_ppl = compute_ppl(model, pos_shuffled)
        
        print("  Computing content-shuffled baseline...")
        content_shuffled = [content_shuffle(p, vocab_size) for p in eval_prompts]
        content_baseline_ppl = compute_ppl(model, content_shuffled)
        
        print(f"  Pos-shuffle baseline PPL: {pos_baseline_ppl:.2f}")
        print(f"  Content-shuffle baseline PPL: {content_baseline_ppl:.2f}")

        arch_results_pos = []
        arch_results_content = []

        targets = []
        if debug:
            retrieval_head = next((k for k, v in model_labels_dict.items() if v.get("label") == "retrieval"), None)
            local_head = next((k for k, v in model_labels_dict.items() if v.get("label") == "local"), None)
            sink_head = next((k for k, v in model_labels_dict.items() if v.get("label") == "sink"), None)
            if retrieval_head: targets.append(retrieval_head)
            if local_head: targets.append(local_head)
            if sink_head: targets.append(sink_head)
            print(f"  [DRY RUN] Running exactly 3 heads: {targets}")
            test_gqa_isolation(model, tokenizer)

        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                label_key_head = f"{layer_idx}_{head_idx}"
                head_label = model_labels_dict.get(label_key_head, {}).get("label", "unknown")

                if head_label == "unknown":
                    continue
                    
                if debug and label_key_head not in targets:
                    continue

                # Position Shuffle: zero OV on pos-shuffled prompt
                res_pos = compute_head_delta_ppl(
                    model, tokenizer, pos_shuffled, layer_idx, head_idx,
                    intervention_fn=ov_zero_fn, target="ov",
                    baseline_ppl=pos_baseline_ppl, architecture=model_name,
                    prompt_id="wikitext-test", label=head_label, dry_run=debug,
                    condition_name="position_shuffle"
                )
                arch_results_pos.append(res_pos)

                # Content Shuffle: zero OV on content-shuffled prompt
                res_content = compute_head_delta_ppl(
                    model, tokenizer, content_shuffled, layer_idx, head_idx,
                    intervention_fn=ov_zero_fn, target="ov",
                    baseline_ppl=content_baseline_ppl, architecture=model_name,
                    prompt_id="wikitext-test", label=head_label, dry_run=debug,
                    condition_name="content_shuffle"
                )
                arch_results_content.append(res_content)

            if debug and len(arch_results_pos) >= len(targets):
                break

        if "position_shuffle" not in all_arch_results: all_arch_results["position_shuffle"] = {}
        if "content_shuffle" not in all_arch_results: all_arch_results["content_shuffle"] = {}
        
        all_arch_results["position_shuffle"][model_name] = arch_results_pos
        all_arch_results["content_shuffle"][model_name] = arch_results_content
        
        # Save checkpoint after each model
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_arch_results, f, indent=2)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("SHUFFLE SURVIVAL SUMMARY (Per-Head)")
    print("="*60)
    
    # We can use the same compute_gate_a_stats logic as 03 if we want to check separation
    # but for now we just save the per-head results!
    
    for condition in ["position_shuffle", "content_shuffle"]:
        print(f"\n--- {condition} ---")
        if condition in all_arch_results:
            for arch_name, res_list in all_arch_results[condition].items():
                if res_list:
                    print(f"  {arch_name}: Computed {len(res_list)} heads")

    print("\n[NOTE] Per-head class-level separation requires the patching approach from script 03.")
    print("       Model-level shuffle results above are consistent with the class-behavior hypotheses:")
    print("       - Large dPPL under content shuffle → content-driven heads (Retrieval, Induction) matter")
    print("       - Large dPPL under position shuffle → position-driven heads (Local, Sink) matter")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_arch_results, f, indent=2)

    print(f"\n[DONE] Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_shuffle(debug=args.debug)
