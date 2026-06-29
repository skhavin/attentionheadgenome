# HeadGenome: The Attention Atlas — Full Execution Plan (v3)
> Author + peer + Claude critique patches applied. This is the canonical, bulletproof roadmap.

---

## Objective

Prove a universal, predictive cross-architecture taxonomy of attention heads ("HeadGenome") and compile it into a training-free Attention Compiler: a system that takes any unseen model, classifies its attention DNA in minutes, and applies optimal hardware approximation with zero calibration.

---

## Existing Assets (old-proj/)

| File | Description |
|---|---|
| `old-proj/outputs/phase1/attention_patterns.pkl` | GPT-2 Medium, **500 docs**, WikiText, relative-offset histograms per (layer, head) |
| `old-proj/outputs/phase4/qwen_attention_patterns.pkl` | Qwen-0.5B, **50 docs** ⚠️ needs re-profile |
| `old-proj/outputs/phase4/meta-llama-3.1-8b-bnb-4bit_attention_patterns.pkl` | Llama-8B 4bit, **50 docs** ⚠️ needs re-profile |
| `old-proj/phase1/run_profiling.py` | GPT-2 profiling pipeline — reuse pattern |
| `old-proj/phase4/profile_llama.py` | Large-model profiling template — reuse |

**Qwen-1.5B: no existing data at all. New profiling run required.**

---

## ⚠️ Critical Metric Corrections (Must Apply Before Any Code)

### Correction 1 — Sink Metric Fix
- **Wrong:** `hist[-50:]` (long-distance tail = wide local head, NOT a BOS anchor)
- **Correct:** `hist[0:4]` — absolute BOS position. Index 0 = distance 0 = first token in sequence.

### Correction 2 — Retrieval = Entropy Collapse, Not Junk Drawer
- **Wrong:** Retrieval = "not local, not sink" (mass in `hist[10:-50]`). Lets diffuse heads pass.
- **Correct:** Retrieval = content-dependent head that **collapses attention entropy** when a matching key appears in a synthetic KV retrieval prompt. Baseline-diffuse heads stay high-entropy on both inputs.

### Correction 3 — 4th Cluster Is Empirical, Never Pre-Labeled
- Do not assume Cluster 4 = "composition heads."
- After clustering, inspect the 4th centroid's mechanistic signature before assigning any name.
- If it shows no distinguishing structure, label it: `"diffuse_background_noise"` — this is a valid and scientifically honest result.

### Correction 4 — 300-Doc Rule (Sample Symmetry)
- Clusters don't converge until ~300 docs (proven by `old-proj/phase1/check_stability.py`).
- GPT-2's 500-doc pkl is truncated to the first 300 indices from `dataset_index.json`.
- Qwen-0.5B and Llama-3.2-1B must be profiled to 300 docs.
- All 4 models compare on the same 300 documents.

### Correction 5 — Precision Symmetry & Pivot to Llama-3.2-1B
- **Quantization Confound:** Comparing a 4-bit model to BF16 models introduces noise and can shift KMeans boundaries (causing the Jaccard similarity sanity check to fail despite 99.999% attention pattern cosine similarity).
- **Fix:** Pivot from Llama-8B (4-bit) to **Llama-3.2-1B** (`unsloth/Llama-3.2-1B`) in native precision (FP16/BF16).
- Llama-3.2-1B fits comfortably in 8GB VRAM in native precision, maintains the official Llama-3 architecture family (RoPE, GQA, SwiGLU), and provides absolute precision symmetry across all profiled models (GPT-2, Qwen-0.5B, Qwen-1.5B, Llama-1B).

---

## Phase 1: Core Taxonomy Isolation

> **Scope:** Prove k=4 taxonomy holds cross-architecturally on a **single domain (WikiText) first**.
> Cross-domain generalization (Phase 1B) only starts after this baseline is solid.

---

### Step 1.0 — Generate Shared Dataset Index
**Script:** `phase1/step1_generate_index.py`

- Load all WikiText-103 train articles using the same logic as `old-proj/data_utils.py`.
- Sample 300 article indices using `random.seed(42)`.
- Save to `outputs/phase1/dataset_index.json`.
- Every subsequent profiling script reads this file. No model ever sees different documents.

**Output:** `outputs/phase1/dataset_index.json`
```json
{
  "seed": 42,
  "dataset": "Salesforce/wikitext",
  "config": "wikitext-103-v1",
  "split": "train",
  "num_docs": 300,
  "indices": [1042, 7, 893, 211, "..."]
}
```

---

### Step 1.1 — Quantization Sanity Check (Complete / Bypassed)
**Script:** `phase1/step2_quant_check.py`

- We completed the quantization check on Qwen-1.5B.
- **Finding:** 4-bit attention histograms have 99.999% cosine similarity to BF16, but KMeans decision boundaries are highly sensitive to small perturbations, resulting in a low cluster-overlap Jaccard similarity.
- **Action:** Guided by this finding, we pivoted to Llama-3.2-1B in native precision, rendering 4-bit quantization unnecessary.

---

### Step 1.2 — Production Profiling (300 Docs, All 4 Models)
**Scripts:** `phase1/step3_profile_gpt2.py`, `phase1/step3_profile_qwen.py`, `phase1/step3_profile_llama.py`

| Model | Size | Old Data | Action |
|---|---|---|---|
| GPT-2 Medium | 345M | 500 docs ✅ | **Re-profile** to 300 indices from `dataset_index.json` |
| Qwen-0.5B | 500M | 50 docs ⚠️ | **Re-profile** to 300 docs |
| Qwen-1.5B | 1.5B | None ❌ | **New profile** to 300 docs |
| Llama-3.2-1B | 1.2B | None ❌ | **New profile** to 300 docs in native FP16/BF16 |

**Each script outputs (both pkl and json):**
- `outputs/phase1/{model_slug}_patterns.pkl` — raw histograms per (layer, head) per doc
- `outputs/phase1/{model_slug}_patterns_summary.json` — per-head mean histogram (list of floats)

---

### Step 1.3 — Clustering & Extreme Head Comparison (The Truth Test)
**Script:** `phase1/step4_extreme_heads.py`

Load all 4 model summaries. For each head compute:

| Metric | Formula | Note |
|---|---|---|
| `sink_score` | `hist[0:4].sum()` | BOS anchor mass (corrected) |
| `local_score` | `hist[1:10].sum()` | Adjacent token mass |
| `retrieval_score` | entropy delta / selective std | Content-dependent collapse |
| `rel_depth` | `layer / (total_layers - 1)` | Normalized 0.0 → 1.0 |

Extract **top-5 per type per model**. Compare relative depths across models.

**Graded Success Criteria:**
| Verdict | Condition |
|---|---|
| 🟢 **Strong Genome** | Top-5 extremes for all 4 models land within ±5% relative depth bands |
| 🟡 **Scale-Dependent Genome** | 3/4 align (e.g., modern RoPE models align, but GPT-2 diverges) |
| 🔴 **Null Hypothesis** | Depths scattered — head placement is stochastic |

**Output:** `outputs/phase1/extreme_heads_comparison.json`
```json
{
  "metadata": {"num_docs": 300, "k": 4, "corrected_sink_metric": "hist[0:4]"},
  "GPT-2": {
    "total_layers": 24,
    "top_sink": [{"layer": 0, "head": 3, "rel_depth": 0.0, "sink_score": 0.91}, "..."],
    "top_local": ["..."],
    "top_retrieval": ["..."]
  },
  "Qwen-0.5B": {},
  "Qwen-1.5B": {},
  "Llama-3.2-1B": {},
  "verdict": "Strong Genome"
}
```

---

### Step 1.4 — Negative Control (~10 min compute)
**Script:** `phase1/step5_negative_control.py`

- Instantiate GPT-2 Medium with **random weights** (`AutoModelForCausalLM.from_config(...)`).
- Run the same profiling pipeline on the same 300-doc index.
- Cluster with k=4. Compare inertia + silhouette vs. trained GPT-2.

**Pass condition:** Random weights → high inertia, low silhouette. Trained → low inertia, high silhouette.

**Output:** `outputs/phase1/negative_control.json`
```json
{
  "trained_gpt2": {"avg_inertia": 0.21, "silhouette": 0.74},
  "random_gpt2":  {"avg_inertia": 1.83, "silhouette": 0.12},
  "verdict": "PASS — clusters are learned, not softmax geometry artifacts"
}
```

---

### Step 1.5 — Characterize Cluster 4 (Empirical)
**Script:** `phase1/step6_characterize_cluster4.py`

- Inspect the 4th centroid's mean histogram shape.
- Check for: shifted token copying, structured position offsets, or uniform diffuse attention.
- Assign exactly one label from: `"induction"` / `"composition"` / `"diffuse_background_noise"`.

**Output:** `outputs/phase1/cluster_characterization.json`
```json
{
  "cluster_4_signature": "moderate distance 20-100, no entropy collapse",
  "label": "diffuse_background_noise",
  "note": "No sharp mechanistic signature found. Labeled as background communication channel — a valid scientific result."
}
```

---

## Phase 1B: Cross-Domain Generalization (Secondary Verification)
> Only run after Steps 1.0–1.5 pass completely.

Profile each model on RULER, Code (Python), and Chat. Map heads into the WikiText-derived k=4 boundaries. Do NOT re-fit — only predict. Measure how many heads retain their WikiText cluster assignment.

**Success:** Same head maps to same cluster across all 4 text regimes → taxonomy is universal.

---

## Phase 2: Chronological Mapping (The Spatial Law)
**Script:** `phase2/step1_depth_trajectory.py`

Aggregate cluster assignments by relative depth bin (0.05 width) across all models.

**Expected pattern:**
- Depth `0.0–0.2` → Sink + Local dominance
- Depth `0.3–0.7` → Retrieval cluster peak
- Depth `0.8–1.0` → Cluster 4 dominance

> **Scope is bounded to 125M–8B parameters.** Any 70B extrapolation is a *future hypothesis only*.

**Output:** `outputs/phase2/depth_trajectory.json`

---

## Phase 3: Weight-Based Prediction (The Genetic Code)
**Script:** `phase3/step1_weight_features.py`

Extract **architecture-agnostic** fixed-size feature vectors for each head. Raw weights break cross-arch (dimension mismatch). Use:

| Feature | Description |
|---|---|
| SVD top-16 singular values of `W_q @ W_k.T` | Fixed-size eigenvalue distribution |
| Weight matrix entropy | `H = -Σ p·log(p)` after softmax normalize |
| Diagonal vs. off-diagonal magnitude ratio | Projection weight anisotropy |
| Frobenius norm ratio `‖W_v‖ / ‖W_q‖` | Value vs. query scale imbalance |

Train lightweight classifier on Phase 1 labels. Evaluate cross-model generalization.

**Output:** `outputs/phase3/weight_features.json` + classifier accuracy

---

## Phase 4: Runtime Compilation (The Infrastructure Engine)
**Script:** `phase4/step1_routing_policy.py`

| Cluster Label | Runtime Approximation |
|---|---|
| Sink | Keep only BOS + last W tokens in KV cache |
| Local | Sliding window: evict tokens outside W-token window |
| Retrieval | Full attention / Heavy Hitter — never evict |
| Cluster 4 (TBD) | Determined from Step 1.5 empirical finding |

**Success:** ≤1% perplexity degradation on unseen model with zero calibration data.

---

## Strict Execution Order

```
Step 1.0  →  phase1/step1_generate_index.py      (lock 300 shared docs)
Step 1.1  →  phase1/step2_quant_check.py          (Qwen-1.5B BF16 vs 4bit, 50 docs)
Step 1.2  →  phase1/step3_profile_*.py             (all 4 models, 300 docs)
Step 1.3  →  phase1/step4_extreme_heads.py         (THE truth test — 20 min)
Step 1.4  →  phase1/step5_negative_control.py      (random weights baseline)
Step 1.5  →  phase1/step6_characterize_cluster4.py (empirical inspection)
Phase 1B  →  cross-domain profiling               (only after all above PASS)
Phase 2   →  phase2/step1_depth_trajectory.py
Phase 3   →  phase3/step1_weight_features.py
Phase 4   →  phase4/step1_routing_policy.py
```
