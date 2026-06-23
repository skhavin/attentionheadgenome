"""
headgenome.benchmarks.speed
───────────────────────────
Measures TTFT, TPOT, E2E latency, and peak VRAM.
Follows NVIDIA NIM / vLLM benchmark conventions:
  - TTFT  = max_new_tokens=1 generation time (pure prefill)
  - TPOT  = (e2e_latency - TTFT) / num_generated_tokens
  - E2E   = full generation latency
  - VRAM  = torch.cuda.max_memory_allocated()
"""

from __future__ import annotations
import time, statistics
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_ttft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str = "cuda",
    warmup: int = 3,
    runs: int = 10,
) -> Dict:
    """
    Measure Time-To-First-Token (TTFT) = pure prefill latency.
    Runs max_new_tokens=1 so almost all time is in forward pass.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**gen_kwargs)
        _sync()

    # Timed runs
    times = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**gen_kwargs)
        _sync()
        times.append(time.perf_counter() - t0)

    mean_s   = statistics.mean(times)
    median_s = statistics.median(times)
    std_s    = statistics.stdev(times) if len(times) > 1 else 0.0
    p90_s    = sorted(times)[int(0.9 * len(times))]

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return {
        "prompt_tokens":      prompt_len,
        "ttft_ms_mean":       mean_s   * 1000,
        "ttft_ms_median":     median_s * 1000,
        "ttft_ms_std":        std_s    * 1000,
        "ttft_ms_p90":        p90_s    * 1000,
        "prefill_tok_s":      prompt_len / mean_s,
        "peak_vram_gb":       peak_vram,
    }


def measure_e2e(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    new_tokens: int = 128,
    device: str = "cuda",
    warmup: int = 2,
    runs: int = 5,
) -> Dict:
    """
    Measure end-to-end latency (TTFT + decode) and derive TPOT.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Measure TTFT separately for accurate TPOT
    ttft_s = measure_ttft(model, tokenizer, prompt, device, warmup=warmup, runs=max(runs, 5))["ttft_ms_mean"] / 1000

    # Warmup E2E
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**gen_kwargs)
        _sync()

    # Timed E2E runs
    times = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**gen_kwargs)
        _sync()
        times.append(time.perf_counter() - t0)

    mean_e2e = statistics.mean(times)
    decode_s = max(mean_e2e - ttft_s, 1e-9)
    tpot_s   = decode_s / new_tokens
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return {
        "prompt_tokens":      prompt_len,
        "generated_tokens":   new_tokens,
        "e2e_latency_ms":     mean_e2e * 1000,
        "ttft_ms":            ttft_s   * 1000,
        "decode_latency_ms":  decode_s * 1000,
        "tpot_ms":            tpot_s   * 1000,
        "decode_tok_s":       new_tokens / decode_s,
        "peak_vram_gb":       peak_vram,
    }


def full_speed_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    new_tokens: int = 128,
    device: str = "cuda",
    warmup: int = 3,
    runs: int = 10,
) -> Dict:
    """Run both TTFT and E2E and merge results."""
    ttft_res = measure_ttft(model, tokenizer, prompt, device, warmup, runs)
    e2e_res  = measure_e2e(model, tokenizer, prompt, new_tokens, device, warmup, min(runs, 5))
    return {**ttft_res, **e2e_res}
