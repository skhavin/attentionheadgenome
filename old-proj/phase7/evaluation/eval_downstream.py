# phase7/eval_downstream.py
#
# Phase 5 — Downstream Task Benchmarks (Not Just PPL)
#
# PPL is least sensitive to a single head misfiring on a rare-but-critical token.
# This script runs two complementary downstream evaluations:
#
#   5a. Repeated-Fact QA (constructed in-process)
#       Template: "[fact]. [1000 tokens filler]. What is [fact_query]?"
#       500 prompts, exact-match accuracy.
#       If Δacc > 0.5%: a Tier 1 head is likely misclassified.
#
#   5b. LM-Eval-Harness (lambada_openai, hellaswag, winogrande)
#       Standard benchmark suite via lm_eval package.
#       Run on full_attention and substitute_with_detector.
#       Target: Δacc ≤ 0.5% on each task.
#
# Usage:
#   python phase7/eval_downstream.py --task repeated_qa
#   python phase7/eval_downstream.py --task lm_eval
#   python phase7/eval_downstream.py --task both

import sys, os, argparse, json, pickle, random, subprocess
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PHASE7_DIR

OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_cache")

FILLER_SENTENCE = (
    "The researchers continued their systematic investigation into the "
    "fundamental properties of the observed phenomena. "
)

# Number of repeated-QA prompts
NUM_REPEATED_QA = 500
# Filler length between the fact and the question
FILLER_TOKENS   = 1000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Downstream task benchmarks")
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--task", choices=["repeated_qa", "lm_eval", "both"],
                   default="both")
    p.add_argument("--modes", nargs="+",
                   default=["full_attention", "substitute_tier1_detector"],
                   choices=["full_attention", "substitute_tier1",
                            "substitute_tier1_detector"])
    p.add_argument("--audit_path",
                   default=os.path.join(PHASE7_DIR, "head_audit.pkl"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--lm_eval_tasks", nargs="+",
                   default=["lambada_openai", "hellaswag", "winogrande"],
                   help="lm-eval task names (default: lambada_openai hellaswag winogrande)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for lm-eval (default: 8)")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    strategies = [
        ({"device_map": {"": device}}, "single-device"),
        ({"device_map": "auto"},       "auto"),
        ({"device_map": "auto", "offload_folder": OFFLOAD_DIR,
          "offload_state_dict": True}, "disk-offload"),
    ]
    for extra, tag in strategies:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                attn_implementation="eager", **extra)
            model.eval()
            print(f"  Loaded [{tag}]")
            return model, tokenizer
        except Exception as e:
            print(f"  Failed [{tag}]: {str(e)[:100]}")
    raise RuntimeError(f"Could not load {model_name}")


# ---------------------------------------------------------------------------
# Tier lists
# ---------------------------------------------------------------------------

def load_tier_lists(audit_path):
    if not os.path.exists(audit_path):
        print(f"  WARNING: {audit_path} not found. Run audit_heads.py first.")
        return [], []
    with open(audit_path, "rb") as f:
        audit = pickle.load(f)
    tier1 = [(r["layer"], r["head"], r["type"]) for r in audit["rows"] if r.get("tier") == 1]
    tier2 = [(r["layer"], r["head"], r["type"]) for r in audit["rows"] if r.get("tier") == 2]
    return tier1, tier2


def apply_mode(model, mode, tier1, tier2):
    if mode == "full_attention":
        return None
    from phase7.regime_detector import RegimeSwitchingPatcher
    if mode == "substitute_tier1":
        patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1, tier2_heads=[])
    else:  # substitute_tier1_detector
        patcher = RegimeSwitchingPatcher(model, tier1_heads=tier1, tier2_heads=tier2)
    return patcher.restore


# ---------------------------------------------------------------------------
# 5a. Repeated-Fact QA
# ---------------------------------------------------------------------------

def _make_filler_text(tokenizer, target_tokens):
    reps = target_tokens // len(tokenizer.encode(FILLER_SENTENCE)) + 3
    text = FILLER_SENTENCE * reps
    ids  = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)


FACT_DB = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
    "Brazil": "Brasilia", "Canada": "Ottawa", "India": "New Delhi",
    "Australia": "Canberra", "Egypt": "Cairo", "Mexico": "Mexico City",
    "Argentina": "Buenos Aires", "Italy": "Rome", "Spain": "Madrid",
    "China": "Beijing", "Russia": "Moscow", "South Korea": "Seoul",
    "Netherlands": "Amsterdam", "Sweden": "Stockholm", "Norway": "Oslo",
    "Denmark": "Copenhagen", "Finland": "Helsinki",
    "Portugal": "Lisbon", "Greece": "Athens", "Poland": "Warsaw",
    "Austria": "Vienna", "Switzerland": "Bern",
}


def build_repeated_qa_prompts(tokenizer, num_prompts=NUM_REPEATED_QA,
                               filler_tokens=FILLER_TOKENS):
    """
    Build 500 prompts of the form:
        "The capital of <Country> is <Capital>. [1000 filler tokens]. What is the capital of <Country>?"
    Exact-match accuracy on the model's first generated token(s).
    """
    rng = random.Random(777)
    facts = list(FACT_DB.items())
    prompts = []

    for i in range(num_prompts):
        country, capital = rng.choice(facts)
        fact_sentence  = f"The capital of {country} is {capital}. "
        question       = f"What is the capital of {country}? The capital is"

        filler = _make_filler_text(tokenizer, filler_tokens)
        text = fact_sentence + filler + question

        prompts.append({
            "text":    text,
            "answer":  capital,
            "country": country,
        })

    return prompts


def run_repeated_qa(model, tokenizer, device, prompts):
    """Run exact-match QA on the repeated-fact prompt set."""
    correct = 0
    total   = 0

    for sample in tqdm(prompts, desc="Repeated-Fact QA"):
        inputs = tokenizer(sample["text"], return_tensors="pt",
                           truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if sample["answer"].lower() in pred.lower():
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    return {"accuracy": acc, "correct": correct, "total": total, "vram_mb": vram}


# ---------------------------------------------------------------------------
# 5b. LM-Eval-Harness
# ---------------------------------------------------------------------------

def run_lm_eval(model_name, mode, tier1, tier2, tasks, device, batch_size, output_dir):
    """
    Run lm_eval via Python API if available, else subprocess fallback.

    The substitution patcher is applied by saving a patched model temporarily.
    For simplicity, this function runs lm_eval in full_attention mode via subprocess
    and produces a comparison table for the substitute modes separately.

    For a full integration with the patcher, use the lm_eval Python API
    and pass a custom model wrapper. See the NOTE below.
    """
    # NOTE: Full integration requires wrapping the patched model as an lm_eval
    # HFLM object. The subprocess approach below runs the standard lm_eval CLI
    # and is the recommended path for getting accurate numbers without fighting
    # the lm_eval model API.

    out_file = os.path.join(output_dir, f"lm_eval_{mode}.json")

    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)

    task_str = ",".join(tasks)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name}",
        "--tasks", task_str,
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", out_file,
        "--log_samples",
    ]

    print(f"  Running lm_eval: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"  lm_eval failed:\n{result.stderr[:500]}")
            return None
        # lm_eval writes to out_file
        if os.path.exists(out_file):
            with open(out_file) as f:
                return json.load(f)
    except FileNotFoundError:
        print("  lm_eval not installed. Install with: pip install lm-eval")
        print("  Skipping lm_eval benchmark.")
        return None
    except subprocess.TimeoutExpired:
        print("  lm_eval timed out (2h limit).")
        return None

    return None


def parse_lm_eval_results(results_json, tasks):
    """Extract per-task accuracy from lm_eval output JSON."""
    if results_json is None:
        return {}
    parsed = {}
    lm_results = results_json.get("results", {})
    for task in tasks:
        if task in lm_results:
            r = lm_results[task]
            # lm_eval uses different metric keys per task
            acc = r.get("acc,none") or r.get("acc_norm,none") or r.get("acc")
            parsed[task] = float(acc) if acc is not None else float("nan")
    return parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(PHASE7_DIR, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(PHASE7_DIR,
                             f"downstream_{args.model.replace('/', '_')}.pkl")

    print(f"\n{'='*72}")
    print(f"  Phase 5 — Downstream Tasks | model={args.model}")
    print(f"  task={args.task} | modes={args.modes}")
    print(f"{'='*72}\n")

    results = {}
    if args.resume and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            results = pickle.load(f)

    tier1, tier2 = load_tier_lists(args.audit_path)

    # ---- 5a: Repeated-Fact QA ----
    if args.task in ("repeated_qa", "both"):
        print("\n[5a] Repeated-Fact QA")
        model, tokenizer = load_model(args.model, str(device))
        device_actual = next(model.parameters()).device
        prompts = build_repeated_qa_prompts(tokenizer)

        for mode in args.modes:
            key = f"repeated_qa_{mode}"
            if key in results:
                r = results[key]
                print(f"  [SKIP] {key}: acc={r['accuracy']:.2%}")
                continue

            print(f"\n  Mode: {mode}")
            restore_fn = apply_mode(model, mode, tier1, tier2)
            r = run_repeated_qa(model, tokenizer, device_actual, prompts)
            if restore_fn:
                restore_fn()

            results[key] = {**r, "mode": mode, "task": "repeated_qa"}
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"  Accuracy: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

        # Summary
        print(f"\n  Repeated-Fact QA Results:")
        baseline_acc = results.get(f"repeated_qa_full_attention", {}).get("accuracy")
        for mode in args.modes:
            key = f"repeated_qa_{mode}"
            if key in results:
                r = results[key]
                delta = (r["accuracy"] - baseline_acc) if baseline_acc else float("nan")
                ok = "✓" if abs(delta) < 0.005 else "✗"
                print(f"    {mode:<35} acc={r['accuracy']:.2%}  Δ={delta:+.2%}  <0.5% {ok}")

        del model  # free VRAM before lm_eval

    # ---- 5b: LM-Eval-Harness ----
    if args.task in ("lm_eval", "both"):
        print("\n[5b] LM-Eval-Harness")
        print(f"  Tasks: {args.lm_eval_tasks}")
        print("  NOTE: lm_eval runs full_attention by default (standard protocol).")
        print("  Substitute-mode lm_eval requires wrapping the patched model as HFLM.")
        print("  See phase7/eval_downstream.py → run_lm_eval() for integration notes.\n")

        for mode in ["full_attention"]:   # run baseline only via subprocess
            key = f"lm_eval_{mode}"
            if key in results:
                print(f"  [SKIP] {key} (checkpoint)")
                continue

            print(f"  Running lm_eval [{mode}]...")
            lm_results_json = run_lm_eval(
                args.model, mode, tier1, tier2,
                args.lm_eval_tasks, str(device),
                args.batch_size, PHASE7_DIR,
            )
            parsed = parse_lm_eval_results(lm_results_json, args.lm_eval_tasks)
            results[key] = {"mode": mode, "task": "lm_eval", "scores": parsed}
            with open(save_path, "wb") as f:
                pickle.dump(results, f)

            if parsed:
                print(f"  LM-Eval results [{mode}]:")
                for task, acc in parsed.items():
                    print(f"    {task:<25} acc={acc:.3f}")

        # Print comparison table (full_attention only until HFLM wrapper is added)
        if f"lm_eval_full_attention" in results:
            full_scores = results["lm_eval_full_attention"].get("scores", {})
            print(f"\n  LM-Eval Accuracy (full_attention baseline):")
            print(f"  {'Task':<25} {'Acc':>8}")
            print(f"  {'-'*35}")
            for task, acc in full_scores.items():
                tgt_ok = "✓" if not (acc != acc) else "—"
                print(f"  {task:<25} {acc:>8.3f}  {tgt_ok}")
            print(f"\n  Target: Δacc ≤ 0.5% between full_attention and substitute modes")
            print(f"  To run substitute-mode lm_eval: implement HFLM wrapper in run_lm_eval()")

    print(f"\n  All results saved to: {save_path}")


if __name__ == "__main__":
    main()
