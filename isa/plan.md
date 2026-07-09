# ISA Unified Methodology Pipeline: Pre-Registered Execution Checklist

**PRE-REGISTRATION LOCK TIMESTAMP:** `2026-07-09T22:00:00Z`
*Any execution of Confirmation Set data prior to this timestamp is formally discarded as contaminated. All thresholds and decision rules below are fixed as of this timestamp.*

## Pre-Flight & Structural Controls
- `[x]` Split `dataset_60.json` into `Discovery (N=40)` and `Confirmation (N=20)`. *(Completed)*
- `[ ]` Ensure testing loops over: `Qwen2.5-1.5B`, `Llama-3.2-1B`, and `Gemma-2-2B` (8-bit).
- `[ ]` Ensure layer indices are dynamically calculated (fractional) to handle Gemma's 26 layers vs Qwen's 28.
- `[ ]` **Global Thresholds Lock:** $|δ| > 0.33$, $p < 0.05$ after **Global FDR correction** (Benjamini-Hochberg) across all hypotheses in all 12 phases × 3 models.
- `[ ]` **Architecture Failure Rule:** If 1 of 3 architectures fails pre-registered thresholds on any phase, it is reported as a "partial/architecture-dependent result." Models are NEVER excluded post-hoc to clean up a finding.

---

## Phase 1: The Life of a Token
*Claim: Residual stream evolves through identifiable stages.* (Descriptive Phase)
- `[ ]` **Test:** Logit-lens + cosine similarity of $r_l$ vs $r_L$ across all Discovery prompts.
- `[ ]` **Metric:** Layer at which `cosine(r_l, r_L) > 0.9`.
- `[ ]` **Control:** Shuffle target tokens (wrong answers) to ensure crossover layer does not falsely reproduce.
- `[ ]` **Stat:** Wilcoxon signed-rank (real vs. shuffled crossover layer).

## Phase 2: Head Dissection (Q/K/V/OV)
*Claim: Heads split cleanly into interpretable circuit roles.*
- `[ ]` **Discovery:** Compute OV and QK metrics on Discovery Set (N=40). Label roles.
- `[ ]` **Test (Confirmation):** Validate the attention-weight effect size on Confirmation Set (N=20).
- `[ ]` **Metric:** Attention weight on causally relevant token vs. mean attention weight (uniform).
- `[ ]` **Control:** Random token position of equal distance (rules out "recent token" bias).
- `[ ]` **Stat:** One-sample t-test (attention - uniform).

## Phase 3: The Birth of a Retrieval
*Claim: Specific MLP layer creates the query feature for the retrieval head.*
- `[ ]` **Discovery:** Identify candidate query-generating MLPs on Discovery Set.
- `[ ]` **Test (Confirmation):** Path patch MLP outputs (clean → corrupted) on Confirmation Set (N=20).
- `[ ]` **Metric:** Logit-diff restoration (%) per layer.
- `[ ]` **Control:** Patch a random non-adjacent layer's MLP output as a placebo.
- `[ ]` **Stat:** Wilcoxon signed-rank of restoration vs placebo, **FDR corrected across all network layers**.

## Phase 4: The Birth of a Word (Logit Attribution)
*Claim: Attention heads gather semantic features, not exact facts.*
- `[ ]` **Test (Confirmation):** Re-run DLA purely on the Confirmation Set (N=20) to ensure uncontaminated stats. Check if heads output the direct factual token.
- `[ ]` **Null-Rate Sanity Check:** Calculate how often *any* head outputs *any* city name at baseline, to ensure 0% isn't confounded by a global lack of city vocabulary predictions.

## Phase 5: Residual Stream Evolution
*Claim: Residual stream moves through distinct geometric regions.*
- `[ ]` **Discovery:** Fit PCA on residual vectors from the Discovery Set. Track variance explained per layer.
- `[ ]` **Test (Confirmation):** Project Confirmation Set vectors into the Discovery PCA space. 
- `[ ]` **Metric:** Silhouette score of task-type clustering (Fact vs Pattern vs NIAH) on the Confirmation projections.
- `[ ]` **Control:** Cluster random Gaussian vectors of same dimensionality.
- `[ ]` **Stat:** Permutation test (shuffle labels 1000x, target > 95th percentile).

## Phase 6: Information Flow Graph
*Claim: Computational edges map specific semantic routing.*
- `[ ]` **Pre-Registration Constraint:** Candidate edges must be pre-registered based on Phase 2/4 head-role outputs prior to path patching. (Exhaustive search is explicitly forbidden due to compute constraints).
- `[ ]` **Test (Confirmation):** Path patch hypothesized edges on Confirmation Set.
- `[ ]` **Metric:** Logit-diff restoration attributable to a single edge.
- `[ ]` **Control:** Patch same-magnitude random noise onto edge.
- `[ ]` **Stat:** Paired t-test (per-edge vs noise-control), Global FDR correction.

## Phase 7: Head Communication (Circuits)
*Claim: Specific heads talk to specific heads.*
- `[ ]` **Pre-Registration Constraint:** Target head-to-head edges must be pre-registered from Phase 6.
- `[ ]` **Test (Confirmation):** Ablate upstream head, check downstream head's output on Confirmation Set.
- `[ ]` **Metric:** Change in downstream OV-output cosine similarity pre/post ablation.
- `[ ]` **Control:** Ablate a random head at the same layer with no hypothesized connection.
- `[ ]` **Stat:** Paired comparison (hypothesized-edge vs control-edge).

## Phase 8: MLP Genome
*Claim: MLPs act as categorized Memory Neurons.*
- `[ ]` **Discovery:** Run DLA on MLPs (Discovery Set). Classify as `Boost-Correct`, `Suppress-RunnerUp`, `Suppress-Distractors`, `Neutral`.
- `[ ]` **Test (Confirmation):** Ablate categorized MLPs on Confirmation Set (N=20).
- `[ ]` **Control:** Compare against ablating a random early-layer MLP.
- `[ ]` **Stat:** Wilcoxon signed-rank (Target MLP ablation vs Control MLP ablation).
- `[ ]` **Fallback Rule:** If validation fails thresholds, label is permanently downgraded to `Neutral` and reported as a taxonomy failure.

## Phase 9: Generation Timeline
*Claim: Temporal ordering exists (Grammar → Concept → Confidence).*
- `[ ]` **Test:** Classify Logit-lens top-1 token into POS categories. Plot category transitions. (Descriptive only; runs on full set as visualization).
- `[ ]` **Metric:** Median layer of first "concept" vs first "answer" token.
- `[ ]` **Control:** Check ordering against a shuffled-layer null model.

## Phase 10: The Residual Language
*Claim: Intermediate residuals can be decoded into human-readable thoughts.*
- `[ ]` **Discovery:** Train the Tuned Lens (per-layer affine probe) on the Discovery Set (N=40).
- `[ ]` **Test (Confirmation):** Compare top-5 interpretability (Perplexity) against raw `lm_head` on the Confirmation Set (N=20).
- `[ ]` **Metric:** Perplexity of Tuned Lens vs Raw Logit Lens vs Actual next-token.
- `[ ]` **Control:** Untrained random affine probe.
- `[ ]` **Stat:** Friedman test (3+ conditions), post-hoc pairwise.

## Phase 11: The Transformer OS
*Claim: Synthesis framework holds across architectures.*
- `[ ]` **Test:** Re-run Phase 2-8 findings across all 3 models (`Qwen`, `Llama`, `Gemma`) using identical pre-locked thresholds.
- `[ ]` **Metric:** % of components mapping to identical OS roles cross-architecture.
- `[ ]` **Stat:** Chi-square test of homogeneity (do role distributions differ significantly across the 3 architectures?). If $p < 0.05$, report as architecture-dependent.

## Phase 12: The Head ISA
*Claim: Heads execute primitive ops (LOAD, SEARCH, COPY, WRITE).*
- `[ ]` **Discovery:** Operationally define primitives. Label heads on Discovery Set.
- `[ ]` **Test (Confirmation):** Predict head behavior on held-out Confirmation Set using assigned primitive label.
- `[ ]` **Metric:** Classification prediction accuracy on Confirmation Set.
- `[ ]` **Control:** Majority-class baseline prediction rate (must be explicitly calculated based on Discovery distribution, e.g., 33%).
- `[ ]` **Stat:** Binomial test (Classification Accuracy vs Baseline).
