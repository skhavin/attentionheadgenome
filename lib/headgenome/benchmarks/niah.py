"""
headgenome.benchmarks.niah
──────────────────────────
Needle-In-A-Haystack synthetic retrieval benchmark.
Uses deterministic exact-match scoring.
"""

from __future__ import annotations
import random
from typing import Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work and live.",
    "The weather today is surprisingly sunny despite the forecast.",
    "A healthy diet consists of fruits, vegetables, and lean proteins.",
    "The city skyline looks beautiful at night when all the lights are on.",
    "Many people find programming to be both challenging and rewarding.",
    "Music has the power to evoke deep emotions and bring people together.",
    "Exploring the outdoors is a great way to relieve stress.",
    "The history of ancient civilizations continues to fascinate historians.",
    "Global markets rallied today following positive economic news.",
    "Deep learning models require significant computational resources.",
    "Space exploration has revealed thousands of exoplanets in our galaxy.",
    "Classical architecture emphasizes symmetry, proportion, and geometry.",
    "The recipe calls for a dash of cinnamon and a cup of sugar.",
]

NEEDLE_TEMPLATES = [
    ("The secret access code to the vault is {value}.", "What is the secret access code to the vault?"),
    ("The hidden activation phrase is {value}.", "What is the hidden activation phrase?"),
    ("The emergency override password is {value}.", "What is the emergency override password?"),
    ("The magic word for the spell is {value}.", "What is the magic word for the spell?"),
]

WORDS = ["OMEGA", "CRIMSON", "ALPHA", "PHOENIX", "ZETA", "DRAGON", "ECLIPSE", "NOVA", "QUANTUM", "SILVER"]


def _haystack(n: int) -> str:
    return " ".join(random.choices(SENTENCES, k=n))


def _build_prompt(depth: float, haystack_sentences: int = 300, seed: int = 0) -> tuple[str, str]:
    random.seed(seed)
    template, question = random.choice(NEEDLE_TEMPLATES)
    value    = random.choice(WORDS) + "-" + str(random.randint(1000, 9999))
    needle   = template.replace("{value}", value)
    expected = value

    idx  = int(haystack_sentences * depth)
    hay1 = _haystack(idx)
    hay2 = _haystack(haystack_sentences - idx)
    prompt = f"{hay1} {needle} {hay2}\n\nQuestion: {question}\nAnswer:"
    return prompt, expected


def run_niah(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int = 20,
    depths: List[float] | None = None,
    haystack_sentences: int = 300,
    device: str = "cuda",
) -> Dict:
    """
    Run NIAH with exact-match scoring.
    Returns overall accuracy, per-depth accuracy, and raw logs.
    """
    if depths is None:
        depths = [0.1, 0.25, 0.5, 0.75, 0.9]

    tests = []
    for depth in depths:
        for i in range(num_samples):
            seed = int(depth * 10000) + i
            prompt, expected = _build_prompt(depth, haystack_sentences, seed)
            tests.append({"depth": depth, "prompt": prompt, "expected": expected})

    hits_by_depth: Dict[float, int] = {d: 0 for d in depths}
    total_by_depth: Dict[float, int] = {d: 0 for d in depths}
    logs = []

    for test in tests:
        inputs = tokenizer(test["prompt"], return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        hit = test["expected"].lower() in generated.lower()

        if hit:
            hits_by_depth[test["depth"]] += 1
        total_by_depth[test["depth"]] += 1

        logs.append({
            "depth": test["depth"],
            "hit": hit,
            "expected": test["expected"],
            "generated": generated,
            "prompt_tokens": prompt_len,
        })

    total_hits  = sum(hits_by_depth.values())
    total_tests = sum(total_by_depth.values())
    overall_acc = total_hits / total_tests if total_tests > 0 else 0.0

    return {
        "overall_accuracy": round(overall_acc, 4),
        "overall_pct":      round(overall_acc * 100, 1),
        "accuracy_by_depth": {
            str(d): round(hits_by_depth[d] / total_by_depth[d], 4)
            for d in depths
        },
        "logs": logs,
    }
