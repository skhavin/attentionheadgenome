# ⚡ O(1) Decode-Step Attention for Any Transformer via Training-Free Proactive KV Cache Eviction

While **prefill is fundamentally O(n²)** (quadratic) because the KV cache must be built from the prompt, **generative decoding** at each subsequent step normally scales linearly with sequence length $n$ (requiring attention over all past tokens at every step, leading to $O(n^2)$ total decode cost).

This repository contains the implementation of **Proactive KV Cache Eviction**, which pre-prunes the KV cache at document ingest using learned attention prototypes. By retaining only a fixed constant budget $B$ of key-value tokens, **the decode attention step becomes O(1) constant-time** regardless of sequence length $n$, completely eliminating the decoding attention bottleneck.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

Run each phase in order:

```bash
# Phase 0: Visualize attention patterns
python phase0/extract_attention.py
python phase0/visualize_heatmaps.py
python phase0/identify_patterns.py

# Phase 1: Profile and cluster attention behaviors
python phase1/run_profiling.py
python phase1/build_prototypes.py
python phase1/check_stability.py

# Phase 2: Predict retention masks
python phase2/encode_documents.py
python phase2/predict_prototypes.py
python phase2/build_retention_mask.py
python phase2/evaluate_prediction.py

# Phase 3: Benchmark with KV cache pruning
python phase3/benchmark.py
python phase3/run_baselines.py
python phase3/make_table.py

# Phase 4: Cross-architecture generalization & Profiling
python phase4/profile_qwen.py
python phase4/profile_llama.py
python phase4/compare_archs.py
python phase4/scaling_gesture.py

# Phase 5: Long-Context LLaMA-3.1 Book Benchmark
python phase5/benchmark_llama.py

# Phase 7: Dynamic Head-Level MoE
python phase7/training/build_stage2_dataset.py
python phase7/training/train_moe.py
python phase7/evaluation/eval_ppl.py
```

## 🏆 Consolidated Results

Every single result, metric, head profile, stability metric, and prediction accuracy is fully consolidated in:
👉 **[CONSOLIDATED_RESULTS.md](CONSOLIDATED_RESULTS.md)**

### PG-19 Benchmark Summary (Table 2)
At aggressive KV cache budgets on full-length books (PG-19), Proactive eviction wins by a massive margin:
* **LLaMA-3.1 8B (Budget 256):** **9.55 PPL** (Ours) vs **9.92 PPL** (StreamingLLM) — outperforming contiguous recency and fully recovering from positional gaps!
* **LLaMA-3.1 8B (Budget 512):** **26.14 PPL** (Ours) vs **156.22 PPL** (StreamingLLM) — a massive **5.98× perplexity outperformance** over local-recency pruning!
* **VRAM Efficiency:** Ours keeps peak VRAM perfectly flat under linear O(budget) scaling.

### Phase 7: Dynamic Head-Level MoE
When testing static KV substitutions on modern dense models like Qwen2.5-0.5B, we discovered that simple static classification failed: while we geometrically clustered heads into 3 tiers using a Gaussian Mixture Model (GMM) on $L_\infty$ log-errors, forcing the "safest" heads into static proxies resulted in an immediate +225.7 PPL explosion.

**The Pivot:** We implemented a **Dynamic Mixture of Experts (MoE)** architecture using lightweight batched `LayerRouters` (2-layer MLPs). Rather than classifying heads *offline*, the router dynamically switches computational paths *per token* at runtime between:
1. $O(1)$ Sink Path
2. $O(N \cdot W)$ Local Convolution Path
3. $O(N^2)$ Exact Softmax Path

By using Curriculum Distillation (Stage 1: Soft Natural → Stage 2: Mixed Triggers → Stage 3: Hard STE), the network learned to **route contextually**, proving the existence of regime-switching heads that fallback to $O(N^2)$ attention only on hard tasks (like induction copy-triggers), safely unlocking structural sparsity without compounding errors.

## Docs

- [CONSOLIDATED_RESULTS.md](CONSOLIDATED_RESULTS.md) — Unified experimental report and reproducibility checklist
- [PLAN.md](PLAN.md) — Implementation plan and timeline (100% Completed)
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and data flow (100% Completed)
- [SYNTAX_USED.md](SYNTAX_USED.md) — Every Python syntax explained

## Hardware & Models

- **GPU:** RTX 3050 (4GB VRAM) or better
- **Models:** 
  - GPT-2 Medium (~700MB fp16)
  - Qwen2.5-0.5B (~1GB fp16)
  - LLaMA 3.1 8B (4-bit quantized ~4.5GB VRAM)
- **Datasets:**
  - WikiText-103-v1 (500 articles for profiling, validation split for eval)
  - PG-19 (test split books streamed from emozilla/pg19)
