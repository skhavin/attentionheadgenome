# phase7/eval_ruler.py
#
# Phase 4 — RULER Long-Context Task Accuracy
#
# PPL is necessary but not sufficient. This script measures whether the model
# can actually use retained information after substitution — a task-level signal
# that catches subtle failures PPL may miss.
#
# Tasks (subset of RULER, implemented offline — no external RULER installation required):
#   niah   — Needle in a Haystack: content heads must retrieve a buried fact
#   vt     — Variable Tracking: tests induction-like heads (find where X was defined)
#   cwe    — Common Words Extraction: tests local+sink heads
#   qa     — End-to-end QA over a long document
#
# Sequence lengths: [512, 1024, 2048, 4096]
# Modes: full_attention, substitute_tier1, substitute_with_detector
#
# Targets:
#   Tier 1 substitution: Δacc < 1%
#   With detector:       Δacc < 3%
#
# Usage:
#   python phase7/eval_ruler.py --tasks niah vt cwe qa --seq_lens 512 1024 2048 4096
#   python phase7/eval_ruler.py --tasks niah --modes full_attention substitute_tier1

import sys, os, argparse, json, pickle, random, re
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
    "The researchers continued their investigation into the properties of "
    "various materials under controlled laboratory conditions. "
)
NUM_SAMPLES = 20   # samples per (task, seq_len, mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 — RULER accuracy benchmark")
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--tasks", nargs="+", default=["niah", "vt", "cwe", "qa"],
                   choices=["niah", "vt", "cwe", "qa"])
    p.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--modes", nargs="+",
                   default=["full_attention", "substitute_tier1",
                            "substitute_tier1_detector"],
                   choices=["full_attention", "substitute_tier1",
                            "substitute_tier1_detector"])
    p.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--audit_path",
                   default=os.path.join(PHASE7_DIR, "head_audit.pkl"))
    p.add_argument("--device", default="cuda")
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
# Filler builder
# ---------------------------------------------------------------------------

def _make_filler(tokenizer, target_tokens):
    reps = target_tokens // len(tokenizer.encode(FILLER_SENTENCE)) + 2
    text = FILLER_SENTENCE * reps
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def make_niah_sample(tokenizer, seq_len, rng):
    """
    Needle in a Haystack: single fact buried at a random position.
    Tests content heads — must attend to a specific distant token.
    """
    fact = rng.randint(10000, 99999)
    needle = f" The magic code is {fact}. "
    question = f" What is the magic code? The magic code is"

    needle_ids   = len(tokenizer.encode(needle,   add_special_tokens=False))
    question_ids = len(tokenizer.encode(question, add_special_tokens=False))
    filler_budget = seq_len - needle_ids - question_ids - 10

    frac = rng.uniform(0.1, 0.9)
    pre_filler  = _make_filler(tokenizer, int(filler_budget * frac))
    post_filler = _make_filler(tokenizer, filler_budget - int(filler_budget * frac))

    text = pre_filler + needle + post_filler + question
    return {"text": text, "answer": str(fact), "task": "niah"}


def make_vt_sample(tokenizer, seq_len, rng):
    """
    Variable Tracking: a value is assigned to a variable; must retrieve it later.
    Tests induction-like heads that can track assignments across long contexts.
    """
    var_name = rng.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma"])
    value = rng.randint(100, 999)
    assignment = f" Let {var_name} = {value}. "
    question = f" What is the value of {var_name}? {var_name} ="

    assignment_ids = len(tokenizer.encode(assignment, add_special_tokens=False))
    question_ids   = len(tokenizer.encode(question,   add_special_tokens=False))
    filler_budget  = seq_len - assignment_ids - question_ids - 10

    frac = rng.uniform(0.1, 0.9)
    pre_filler  = _make_filler(tokenizer, int(filler_budget * frac))
    post_filler = _make_filler(tokenizer, filler_budget - int(filler_budget * frac))

    text = pre_filler + assignment + post_filler + question
    return {"text": text, "answer": str(value), "task": "vt"}


def make_cwe_sample(tokenizer, seq_len, rng):
    """
    Common Words Extraction: the most frequent word in a long passage.
    Tests local + sink heads — common words appear throughout the context.
    """
    # Build a document where one word appears significantly more often
    word_pool = ["apple", "banana", "cherry", "date", "elderberry",
                 "fig", "grape", "honeydew", "kiwi", "lemon"]
    target_word = rng.choice(word_pool)
    other_words = [w for w in word_pool if w != target_word]

    # Rough token budget
    text_budget = seq_len - 30  # leave room for question
    question = f" What is the most common word in the passage? The most common word is"
    question_ids = len(tokenizer.encode(question, add_special_tokens=False))
    text_budget -= question_ids

    # Build text: target word appears ~40% of the time, others split the rest
    words = []
    for _ in range(text_budget // 2):  # approximate
        if rng.random() < 0.40:
            words.append(target_word)
        else:
            words.append(rng.choice(other_words))
    text = " ".join(words) + question

    return {"text": text, "answer": target_word, "task": "cwe"}


def make_qa_sample(tokenizer, seq_len, rng):
    """
    End-to-end QA: a fact followed by a long filler passage, then a question.
    End-to-end test of all head types working together.
    """
    capitals = {
        "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
        "Brazil": "Brasilia", "Canada": "Ottawa", "India": "New Delhi",
        "Australia": "Canberra", "Egypt": "Cairo", "Mexico": "Mexico City",
        "Argentina": "Buenos Aires",
    }
    country, capital = rng.choice(list(capitals.items()))
    fact = f" The capital of {country} is {capital}. "
    question = f" What is the capital of {country}? The capital is"

    fact_ids     = len(tokenizer.encode(fact,     add_special_tokens=False))
    question_ids = len(tokenizer.encode(question, add_special_tokens=False))
    filler_budget = seq_len - fact_ids - question_ids - 10

    pre_filler  = _make_filler(tokenizer, int(filler_budget * 0.3))
    post_filler = _make_filler(tokenizer, filler_budget - int(filler_budget * 0.3))

    text = pre_filler + fact + post_filler + question
    return {"text": text, "answer": capital, "task": "qa"}


TASK_GENERATORS = {
    "niah": make_niah_sample,
    "vt":   make_vt_sample,
    "cwe":  make_cwe_sample,
    "qa":   make_qa_sample,
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, text, device, max_new=15):
    """Greedy decode up to max_new tokens after the prompt."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=8192).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def score_answer(pred, answer):
    """Exact-match (case-insensitive): does answer appear in prediction?"""
    return answer.lower() in pred.lower()


def run_task_eval(model, tokenizer, device, task, seq_len, num_samples, seed=42):
    rng   = random.Random(seed)
    gen   = TASK_GENERATORS[task]
    correct = total = 0

    for _ in tqdm(range(num_samples), desc=f"{task} sl={seq_len}", leave=False):
        sample = gen(tokenizer, seq_len, rng)
        pred   = generate_answer(model, tokenizer, sample["text"], device)
        if score_answer(pred, sample["answer"]):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    return {"accuracy": acc, "correct": correct, "total": total, "vram_mb": vram}


# ---------------------------------------------------------------------------
# Mode patching (same as eval_ppl.py)
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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(PHASE7_DIR, exist_ok=True)

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(PHASE7_DIR,
                             f"ruler_{args.model.replace('/', '_')}.pkl")

    print(f"\n{'='*72}")
    print(f"  Phase 4 — RULER Accuracy | model={args.model}")
    print(f"  tasks={args.tasks} | seq_lens={args.seq_lens} | modes={args.modes}")
    print(f"{'='*72}\n")

    results = {}
    if args.resume and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"  Resuming: {list(results.keys())}")

    model, tokenizer = load_model(args.model, str(device))
    device = next(model.parameters()).device
    tier1, tier2 = load_tier_lists(args.audit_path)

    for seq_len in args.seq_lens:
        for task in args.tasks:
            for mode in args.modes:
                key = f"{mode}_{task}_sl{seq_len}"
                if key in results:
                    r = results[key]
                    print(f"  [SKIP] {key}: acc={r['accuracy']:.2%}")
                    continue

                print(f"\n--- {mode} | {task} | sl={seq_len} ---")
                restore_fn = apply_mode(model, mode, tier1, tier2)
                r = run_task_eval(model, tokenizer, device, task, seq_len,
                                  args.num_samples)
                if restore_fn:
                    restore_fn()

                results[key] = {**r, "mode": mode, "task": task,
                                "seq_len": seq_len, "model": args.model}
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"  Accuracy: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print(f"  RULER Accuracy Summary (higher is better)")
    print(f"{'='*80}")
    print(f"  {'Mode':<35} {'Task':<6} {'SeqLen':>7} {'Acc':>7} {'Δacc':>8} {'Target':>8}")
    print(f"  {'-'*75}")

    for seq_len in args.seq_lens:
        for task in args.tasks:
            baseline_key = f"full_attention_{task}_sl{seq_len}"
            baseline_acc = results.get(baseline_key, {}).get("accuracy")
            for mode in args.modes:
                key = f"{mode}_{task}_sl{seq_len}"
                if key not in results:
                    continue
                r = results[key]
                acc = r["accuracy"]
                delta = (acc - baseline_acc) if baseline_acc is not None else float("nan")
                tgt = 0.01 if "detector" not in mode else 0.03
                ok = "✓" if abs(delta) < tgt else "✗"
                print(f"  {mode:<35} {task:<6} {seq_len:>7} "
                      f"{acc:>7.2%} {delta:>+8.2%} {f'<{tgt:.0%} {ok}':>8}")
        print(f"  {'-'*75}")

    print(f"\n  Results saved to: {save_path}")

    # ---- Append to summary.json ----
    summary_path = os.path.join(PHASE7_DIR, "summary.json")
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        summary = {}
    ruler_section = summary.get("ruler", {})
    for seq_len in args.seq_lens:
        for task in args.tasks:
            baseline_key = f"full_attention_{task}_sl{seq_len}"
            base_acc = results.get(baseline_key, {}).get("accuracy")
            for mode in args.modes:
                key = f"{mode}_{task}_sl{seq_len}"
                if key in results:
                    r = results[key]
                    ruler_section[key] = {
                        "acc":   round(r["accuracy"], 4),
                        "delta": round(r["accuracy"] - base_acc, 4) if base_acc else None,
                        "model": args.model,
                    }
    summary["ruler"] = ruler_section
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Headlines appended to: {summary_path}")


if __name__ == "__main__":
    main()
