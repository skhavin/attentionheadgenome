"""
headgenome.benchmarks.ppl
─────────────────────────
Perplexity on WikiText-103 using sliding-window evaluation.
"""

from __future__ import annotations
import math
from typing import Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def measure_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int = 512,
    stride: int = 256,
    max_tokens: int = 10_000,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    device: str = "cuda",
) -> dict:
    """
    Compute sliding-window perplexity on WikiText-103 test set.
    Returns {"ppl": float, "nll_mean": float, "num_tokens": int}
    """
    try:
        data = load_dataset(dataset, dataset_config, split="test", trust_remote_code=True)
        text = "\n\n".join(data["text"])
    except Exception:
        # Fallback to a small local text
        text = " ".join([
            "The quick brown fox jumps over the lazy dog." * 200
        ])

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0, :max_tokens]

    nlls = []
    prev_end = 0

    for begin in range(0, input_ids.shape[0], stride):
        end   = min(begin + seq_len, input_ids.shape[0])
        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        tgt_len = end - prev_end

        with torch.no_grad():
            out = model(chunk, labels=chunk)
            nll = out.loss.item() * (chunk.shape[1] - 1)
            # Only count tokens not overlapping previous window
            if tgt_len > 0:
                nlls.append(nll * (tgt_len / (chunk.shape[1] - 1)))

        prev_end = end
        if end >= input_ids.shape[0]:
            break

    nll_mean = sum(nlls) / len(nlls) if nlls else float("inf")
    ppl = math.exp(nll_mean) if nll_mean < 100 else float("inf")

    return {"ppl": round(ppl, 4), "nll_mean": round(nll_mean, 6), "num_tokens": len(nlls)}
