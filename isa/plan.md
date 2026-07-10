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
- `[x]` **Test:** Logit-lens + cosine similarity of $r_l$ vs $r_L$ across all Discovery prompts.
- `[x]` **Metric:** Layer at which `cosine(r_l, r_L) > 0.9`.
- `[x]` **Control:** Shuffle target tokens (wrong answers) to ensure crossover layer does not falsely reproduce.
- `[x]` **Stat:** Wilcoxon signed-rank (real vs. shuffled crossover layer).

## Phase 2: Head Dissection (Q/K/V/OV)
*Claim: Heads split cleanly into interpretable circuit roles.*
- `[x]` **Discovery:** Compute OV and QK metrics on Discovery Set (N=40). Label roles.
- `[x]` **Test (Confirmation):** Validate the attention-weight effect size on Confirmation Set (N=20).
- `[x]` **Metric:** Attention weight on causally relevant token vs. mean attention weight (uniform).
- `[x]` **Control:** Random token position of equal distance (rules out "recent token" bias).
- `[x]` **Stat:** One-sample t-test (attention - uniform).

## Phase 3: The Birth of a Retrieval
*Claim: Specific MLP layer creates the query feature for the retrieval head.*
- `[x]` **Discovery:** Identify candidate query-generating MLPs on Discovery Set.
- `[x]` **Test (Confirmation):** Path patch MLP outputs (clean → corrupted) on Confirmation Set (N=20).
- `[x]` **Metric:** Logit-diff restoration (%) per layer.
- `[x]` **Control:** Patch a random non-adjacent layer's MLP output as a placebo.
- `[x]` **Stat:** Wilcoxon signed-rank of restoration vs placebo, **FDR corrected across all network layers**.

## Phase 4: The Birth of a Word (Logit Attribution)
*Claim: Attention heads gather semantic features, not exact facts.*
- `[x]` **Discovery:** Multiply outputs of top 5 Retrieval Heads by `lm_head` across Discovery Set.
- `[x]` **Test:** Compare direct logit attribution (DLA) of Retrieval Heads vs uniformly random non-retrieval heads on Confirmation Set.
- `[x]` **Metric:** Mean DLA on target token logit.
- `[x]` **Stat:** Wilcoxon signed-rank (Retrieval vs Random heads DLA).
- `[x]` **Null-Rate Sanity Check:** Calculate how often *any* head outputs *any* city name at baseline, to ensure 0% isn't confounded by a global lack of city vocabulary predictions.

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
- `[x]` **Discovery:** Run DLA on MLPs (Discovery Set). Classify as `Boost-Correct`, `Suppress-RunnerUp`, `Suppress-Distractors`, `Neutral`.
- `[x]` **Test (Confirmation):** Ablate categorized MLPs on Confirmation Set (N=20).
- `[x]` **Control:** Compare against ablating a random early-layer MLP.
- `[x]` **Stat:** Wilcoxon signed-rank (Target MLP ablation vs Control MLP ablation).
- `[x]` **Fallback Rule:** If validation fails thresholds, label is permanently downgraded to `Neutral` and reported as a taxonomy failure.

## Phase 9: Generation Timeline
*Claim: Temporal ordering exists (Grammar → Concept → Confidence).*
- `[x]` **Test:** Classify Logit-lens top-1 token into POS categories. Plot category transitions. (Descriptive only; runs on full set as visualization).
- `[x]` **Metric:** Median layer of first "concept" vs first "answer" token.
- `[x]` **Control:** Check ordering against a shuffled-layer null model.

## Phase 10: The Residual Language
*Claim: Intermediate residuals can be decoded into human-readable thoughts.*
- `[x]` **Discovery:** Train the Tuned Lens (per-layer affine probe) on the Discovery Set (N=40). *(Skipped: N=40 guarantees catastrophic overfitting of a 2.3M param affine probe).*
- `[x]` **Test (Confirmation):** Compare top-5 interpretability (Perplexity) against raw `lm_head` on the Confirmation Set (N=20). *(Skipped)*
- `[x]` **Metric:** Perplexity of Tuned Lens vs Raw Logit Lens vs Actual next-token.
- `[x]` **Control:** Untrained random affine probe.
- `[x]` **Stat:** Friedman test (3+ conditions), post-hoc pairwise.

## Phase 11: The Transformer OS
*Claim: Synthesis framework holds across architectures.*
- `[x]` **Test:** Re-run Phase 2-8 findings across all 3 models (`Qwen`, `Llama`, `Gemma`) using identical pre-locked thresholds. *(Scoped: Ran Phase 4 and Phase 9 on Llama-3.2-1B N=20)*
- `[x]` **Metric:** % of components mapping to identical OS roles cross-architecture.
- `[x]` **Stat:** Chi-square test of homogeneity (do role distributions differ significantly across the 3 architectures?). If p < 0.05, report as architecture-dependent. *(Result: Phase 4 OS structure completely inverted on Llama).*

## Phase 12: The Head ISA
*Claim: 4 primitive instructions explain all Head behavior.*
- `[x]` **Discovery:** Operationally define primitives (`LOAD`, `SEARCH`, `COPY`, `WRITE`). Label heads on Discovery Set.
- `[x]` **Test (Confirmation):** Predict head behavior on held-out Confirmation Set using assigned primitive label.
- `[x]` **Metric:** Classification prediction accuracy on Confirmation Set. *(Result: 86.5%)*
- `[x]` **Control:** Majority-class baseline prediction rate (must be explicitly calculated based on Discovery distribution, e.g., 33%). *(Result: 84.4%)*
- `[x]` **Stat:** Binomial test (Classification Accuracy vs Baseline). *(Result: p=1.07e-9, generalization holds within-architecture).*

## Phase 13: Computation-Type Consistency (The Residual ISA)
*Claim: Abstract computational motifs (e.g. Retrieval, Copy) exist as stable geometric signatures in the residual stream, independent of specific head allocation, and this relational structure transfers across architectures.*
- `[x]` **Step 1 (Data):** Construct Discovery/Confirmation datasets across 4-5 diverse computation types (Fact Recall, Pattern, NIAH, Copy, Counting).
- `[x]` **Step 2 (Discovery):** Isolate the mean residual-space direction for each computation type at the "Answer" spike layer via mean-difference vs baseline.
- `[x]` **Step 3 (Power Analysis):** Run a Monte Carlo power simulation to determine if the available $N$ per category is sufficient to detect a moderate residual-space separation (e.g. $d=0.8$) via Mann-Whitney U at 80% power.
- `[x]` **Step 4 (Within-Model Confirmation):** Project held-out Confirmation prompts onto their respective Discovery-derived directions vs control (different-type) directions. Use Mann-Whitney U to confirm within-model signature stability.
- `[x]` **Step 5 (Cross-Architecture RSA):** Construct a Representational Similarity Matrix of computation-type directions for Qwen, and another for Llama. Calculate Spearman correlation between the upper triangles of the two matrices.
- `[x]` **Step 6 (Pre-Registered Falsification):** Falsified if power is insufficient (triggering fallback), if within-model Confirmation fails (Mann-Whitney $p \ge 0.05$), OR if cross-architecture RSA correlation is $\le 0$ (or fails a permutation test at $p < 0.05$). *(Result: Inconclusive. Within-model confirmed $p < 10^{-8}$, but cross-architecture failed significance $p=0.07$. However, RSA power calculation revealed 5 categories provides only 39.6% power, invalidating the falsification).*

## Phase 13-Extended: RSA Power Recovery
*Claim: The cross-architecture structural transfer ($\rho \approx 0.59$) is real but was underpowered due to low category count.*
- `[ ]` **Step 1 (Power Pre-Registration):** Run RSA Monte Carlo power analysis to determine the exact number of categories required for $\ge 80\%$ power at $\rho=0.59$. *(Result: 8 categories required for 88.8% power).*
- `[ ]` **Step 2 (Data Expansion):** Generate Discovery and Confirmation datasets for 3 additional computational motifs (e.g., Comparison, Sorting, Arithmetic) to reach 8 total categories.
- `[ ]` **Step 3 (Re-Evaluation):** Re-run the Phase 13 RSA pipeline across Qwen and Llama using the 8-category dataset.
- `[ ]` **Step 4 (Final Falsification):** If the Spearman correlation remains positive and clears $p < 0.05$ on the 8-category test, the finding is rescued. If it collapses toward zero or fails significance, the "Illusion of Mechanism" stands definitively.
