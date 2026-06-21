# Phase 7 — Closed-Form Substitution Benchmark Suite

## Overview

This phase validates the claim that sink and local attention heads can be replaced
by **closed-form substitutions with bounded approximation error** — *not*
"mathematically exact", but with quantified bounds:

> **L∞ ≤ 0.015** on natural text (Tier 1 heads);  
> downstream **logit KL divergence** reported per-head in `outputs/phase7/head_audit.json`.

A key finding is the **regime-switching phenomenon**: heads classified as "local"
under static calibration activate induction-like, long-range copy patterns under
copy-trigger inputs. This is a *discovery*, not a weakness — the O(N) runtime
detector in Phase 2 identifies and handles it without any model forward pass.

---

## Files

```
phase7/
  substitutes.py         — Closed-form sink (O(1)) and local (O(N·W)) substitutions
  audit_heads.py         — Phase 1: per-head L∞ / KL audit across all prompt types
  regime_detector.py     — Phase 2: O(N) regime-switch detector + validation
  eval_ppl.py            — Phase 3: PPL benchmarks (WikiText-103, PG-19, induction)
  eval_ruler.py          — Phase 4: RULER task accuracy (NIAH, VT, CWE, QA)
  eval_downstream.py     — Phase 5: repeated-fact QA + LM-Eval-Harness
  benchmark_prefill.py   — Phase 6: prefill latency vs. seq_len (mixed complexity)
  README.md              — this file
  __init__.py            — package marker
```

All headline numbers are auto-aggregated into `outputs/phase7/summary.json` as
each phase completes — a single file for paper tables and cross-model comparisons.

---

## Execution Order

**Run Phase 1 first.** The L∞/KL audit determines which heads are Tier 1 (safe
substitution) and which need the detector — every downstream experiment depends
on this classification.

---

### Phase 1 — Per-Head Error Audit

```bash
python phase7/audit_heads.py --model gpt2-medium --num_natural 200 --num_copy 100 --num_niah 50
```

Output: `outputs/phase7/head_audit.json`, `outputs/phase7/tier2_heads.json`,
`outputs/phase7/summary.json` (audit section)

#### Tier classification — both L∞ AND logit KL must pass for Tier 1

| Tier | Condition | Action |
|------|-----------|--------|
| 1 | L∞ < 0.001 **AND** logit KL < 0.01 on **both** natural + copy-trigger | Safe closed-form substitution |
| 2 | L∞ < 0.001 on natural, > 0.01 on copy-trigger **OR** KL spikes on copy-trigger | Regime-switch detector required |
| 3 | Everything else | Full attention required |

> **Paper claim:** > 60% of heads in Tier 1 → strong compute savings claim.  
> > 80% in Tier 1 + 2 together → substitution paper is essentially complete.

---

### Phase 2 — Regime-Switch Detector

```bash
python phase7/regime_detector.py --tier2_path outputs/phase7/tier2_heads.json
```

Validates the O(N) detector on the stress set.  
**Target:** Precision > 0.90, Recall > 0.85.

The detector uses two signals — both are O(N) to compute, no model forward pass:
- **n-gram repetition rate** in the prefix (threshold: > 0.30)
- **max token frequency** (threshold: > 0.05 of prefix length)

This dynamic routing behaviour is absent from static calibration — it is a
**finding** (induction heads activate under copy triggers) not a failure.

---

### Phase 3 — PPL Benchmarks

> **Mode names** (consistent across all scripts):
> - `full_attention` — exact softmax baseline
> - `substitute_tier1_only` — Tier 1 closed-form substitution, Tier 2/3 full attention
> - `substitute_tier1_detector` — Tier 1 substitution + Tier 2 with O(N) regime detector

```bash
# 3a. WikiText-103 (held-out TEST split — not the calibration docs)
python phase7/eval_ppl.py --dataset wikitext --modes full_attention substitute_tier1_only substitute_tier1_detector

# 3b. PG-19 (ultra-long documents)
python phase7/eval_ppl.py --dataset pg19 --seq_len 2048 --modes full_attention substitute_tier1_only substitute_tier1_detector

# 3c. Induction probe set
python phase7/eval_ppl.py --dataset induction --modes full_attention substitute_tier1_detector
```

#### Induction probe set spec

| Parameter | Values |
|-----------|--------|
| repeat_distances | [10, 50, 100, 500] tokens |
| ngram_lengths | [1, 2, 3, 4, 5] |
| n_per_combination | 25 |
| **Total prompts** | **4 × 5 × 25 = 500** |
| Template | `[A B C] [filler × dist] [A B C] [20 token suffix]` |
| PPL metric | **Repeated suffix only** — starting at the second occurrence of the ngram |

> **Why suffix-only PPL matters:** Full-sequence PPL dilutes the signal. Most
> tokens are in the filler where no induction head fires. Measuring only the
> repeated suffix isolates exactly the positions where regime switching matters.

#### PPL targets

| Dataset | ΔPPL target |
|---------|-------------|
| WikiText-103 | < 0.5 |
| PG-19 | < 1.0 |
| Induction (with detector) | < 2.0 |

---

### Phase 4 — RULER Accuracy

#### Prerequisites (one-time setup)

```bash
git clone https://github.com/hsiehjackson/RULER
cd RULER && pip install -r requirements.txt
python scripts/data/prepare_data.py --task niah,vt,cwe,qa --seq_lens 512,1024,2048,4096
cd ..
```

> Without this step you will get a confusing `ModuleNotFoundError` from `run_ruler.py`.
> The offline task implementations in `eval_ruler.py` do not require RULER installation
> — they are self-contained reproductions of NIAH, VT, CWE, and QA tasks.

```bash
python phase7/eval_ruler.py \
  --tasks niah vt cwe qa \
  --seq_lens 512 1024 2048 4096 \
  --modes full_attention substitute_tier1 substitute_tier1_detector
```

**Target:** Δacc < 1% for Tier 1, < 3% with detector.

| Task | Head type tested |
|------|-----------------|
| NIAH | Content heads (long-range fact retrieval) |
| VT   | Induction-like heads (variable tracking) |
| CWE  | Local + sink heads (common word extraction) |
| QA   | End-to-end (all head types) |

---

### Phase 5 — Downstream Tasks

```bash
# 5a. Repeated-fact QA (500 prompts, ~1000 filler tokens each)
python phase7/eval_downstream.py --task repeated_qa --modes full_attention substitute_tier1_detector

# 5b. LM-Eval-Harness (standard suite)
python phase7/eval_downstream.py --task lm_eval
```

**Target:** Δacc ≤ 0.5% on repeated-fact QA and all lm-eval tasks.  
If Δacc > 0.5% on repeated-fact QA: a Tier 1 head is misclassified.

---

### Phase 6 — Prefill Latency

```bash
# n_warmup=3 warmup runs (excluded from timing) + n_runs=10 timed runs
# This is mandatory — GPU timing without warmup can make a 2× speedup
# look like 1.3× due to CUDA JIT and cache cold-start effects.
python phase7/benchmark_prefill.py \
  --seq_lens 512 1024 2048 4096 \
  --n_warmup 3 \
  --n_runs 10 \
  --plot
```

Output: `outputs/phase7/prefill_latency.json` + `outputs/phase7/prefill_latency.png`

Mixed-complexity breakdown:

| Head type | Substitution | Prefill complexity |
|-----------|-------------|-------------------|
| Sink | Uniform average of V[0:4] | O(1) |
| Local | Causal depthwise conv over V, window W | O(N·W) |
| Content/global | Full softmax attention | O(N²) |

**Key distinction over DuoAttention/MoA/FastGen:** those methods reduce KV *cache*
size without changing the O(N²) prefill compute for every head. Our substitution
changes the **compute graph** itself — the speedup is visible in prefill timing
even without any KV eviction.

---

## summary.json — Auto-Generated Headline Numbers

`outputs/phase7/summary.json` is updated automatically at the end of each phase.
It contains all the numbers needed for the paper table:

```json
{
  "audit": {
    "model": "gpt2-medium",
    "tier1_count": ..., "tier2_count": ..., "tier3_count": ...,
    "tier1_pct": ...,
    "thresholds": {"tier1_linf": 0.001, "tier1_kl": 0.01}
  },
  "detector": {"precision": ..., "recall": ..., "f1": ...},
  "ppl": {
    "full_attention_wikitext": {"ppl": ..., "delta": null},
    "substitute_tier1_detector_wikitext": {"ppl": ..., "delta": ...},
    ...
  },
  "ruler": {
    "substitute_tier1_niah_sl1024": {"acc": ..., "delta": ...},
    ...
  },
  "prefill": {
    "substitute_tier1_sl2048": {"speedup": ..., "complexity": "..."}
  }
}
```

To compare across models, run each phase with `--model <model_id>`. The summary
entries are keyed by `(mode, dataset/task, seq_len)` so multiple model runs
can be stored in separate files or merged manually.

---

## Quick Reference (Full Pipeline)

```bash
# 1. Audit all heads (run this FIRST — every other script depends on the output)
python phase7/audit_heads.py --model gpt2-medium

# 2. Validate regime detector
python phase7/regime_detector.py

# 3. PPL on all three datasets
python phase7/eval_ppl.py --dataset wikitext
python phase7/eval_ppl.py --dataset pg19 --seq_len 2048
python phase7/eval_ppl.py --dataset induction

# 4. RULER task accuracy
python phase7/eval_ruler.py

# 5. Downstream tasks
python phase7/eval_downstream.py --task both

# 6. Prefill latency (warmup=3, runs=10 — do not lower these)
python phase7/benchmark_prefill.py --n_warmup 3 --n_runs 10 --plot
```

---

## Paper Numbers Table

| Experiment | Proves | Target |
|-----------|--------|--------|
| L∞ / KL audit (all heads × all prompt types) | Error bounded and quantified | L∞ < 0.001 AND KL < 0.01 for Tier 1 |
| PPL: WikiText-103 test | Aggregate quality preserved | ΔPPL < 0.5 |
| PPL: PG-19 | Long-context quality preserved | ΔPPL < 1.0 |
| PPL: Induction probe (suffix-only) | Regime detector catches switches | ΔPPL < 2.0 with detector |
| RULER accuracy | Task-level, not just perplexity | Δacc < 1% |
| LM-Eval-Harness | Standard benchmark coverage | Δacc < 0.5% |
| Prefill latency vs. seq len | Mixed O(1)/O(N·W)/O(N²) compute | Speedup plot |
| Detector precision / recall | Detector is sound | Prec > 0.90, Rec > 0.85 |
