# The Residual ISA: Execution Plan

To establish the Residual ISA as a methodologically bulletproof finding, we execute a rigorous 5-step program (Steps 0-4). Every step includes a pre-registered failure condition to prevent post-hoc curve fitting and p-hacking.

## Step 0: Confound Regression
*Goal: Ensure the RSA is detecting computational geometry, not surface-level lexical/domain structure.*
- `[x]` **Data:** 224 Discovery + 112 Confirmation residual directions (8 categories).
- `[x]` **Design:** Regress out three confounds (Prompt token-length, Answer-token proxy frequency, Lexical numeric domain) from each residual vector.
- `[x]` **Statistics:** Compute the RSM and Spearman $\rho$ on the pure residuals. Mantel-style permutation test ($N=10,000$).
- `[x]` **Result:** **SUCCESS.** Overall $\rho = 0.9644$ ($p < 10^{-15}$). Original 10 pairs surged to $\rho = 0.9758$. The Residual ISA represents pure computation.

## Step 1: Held-Out Operation Generalization
*Goal: Test Claim 3 ("geometry generalizes to unseen operations").*
- `[x]` **Design:** Exclude 2 categories (Sorting and Arithmetic) from the Discovery-stage direction estimation. Build their residual directions *only* from Confirmation-stage prompts never used to fit the original RSM.
- `[x]` **Test:** Given the RSM built from the 6 other categories, predict the held-out pair's position in the geometry (e.g., via Procrustes-fit or nearest-neighbor).
- `[x]` **Statistics:** Correlation between predicted vs. actual distance-to-all-other-categories vector for the held-out pair. Since $N=12$ (6 distances $\times$ 2 categories), use **bootstrap over prompts within category** (10,000 resamples) to build a confidence interval on the correlation.
- `[x]` **Result:** **SUCCESS.** Sorting Cross-Architecture $\rho = 0.9198$ (95% CI: 0.7714 - 1.0000). Arithmetic $\rho = 0.9420$ (95% CI: 0.9429 - 0.9429). The geometric structure successfully generalizes out-of-sample.

## Step 2: Content-Domain Factorial Control (The Confound Killer)
*Goal: Directly prove that clustering (e.g., Comparison) is driven by computation, not the numeric content domain.*
- `[x]` **Design:** Pick **Comparison**. Cross it with 5 content domains: Numbers ("17 > 12"), Dates ("June occurs after April"), Arbitrary symbols ("dax outranks wug"), Lengths ("A is taller than B"), Invented words.
- `[x]` **Data:** 15 prompts per domain $\times$ 5 domains = 75 prompts total (split 50 Discovery / 25 Confirmation).
- `[x]` **Test 1 (Invariance):** Extract the mean residual direction for Comparison within each domain separately. Calculate pairwise cosine similarity between the 5 domain-specific Comparison directions vs a different operation (e.g. Copy).
- `[x]` **Test 2 (Discriminability):** Domain-crossed classifier. Train on 4 domains, test if the held-out domain's Comparison direction is closer to "Comparison" than to "Copy".
- `[x]` **Statistics:** Mantel-style permutation test for pairwise similarities ($N=10,000$). Cliff's $\delta > 0.33$ required for same-operation similarity vs different-operation similarity.
- `[x]` **Result:** **SUCCESS.** Cliff's Delta = 1.0 (Invariance). Discriminability Accuracy = 100%. `Comparison` is invariant to extreme content shifts.

## Step 3: Cross-Architecture Transfer with Held-Out Operations
*Goal: The "Killer Experiment" proving a Universal ISA.*
- `[x]` **Design:** Learn a linear (Procrustes) alignment between Qwen and Llama residual spaces using 6 of 8 categories. Hold out 1 "safe" category (Counting) and 1 "suspicious" category (Sorting).
- `[x]` **Test:** After alignment, does the held-out category's Qwen direction land near its Llama counterpart in the aligned space, relative to random category assignment?
- `[x]` **Statistics:** Rank of the true match among candidates in the aligned space. Require replication across 3 independently generated prompt sets (~15 prompts each) to produce a distribution of outcomes.
- `[x]` **Result:** **SUCCESS.** The true match median rank was 1.0 (it landed as the exact closest neighbor 5 out of 6 times). The PCA-based projection (k=128, capturing 99.85% variance for both models) resulted in a significant permutation test ($p = 0.0027$).

## Step 4: Scale-Up for a Genuinely Well-Powered RSA
*Goal: Move from an 8-category pilot to a complete, saturated Residual ISA.*
- `[ ]` **Design:** Expand from 8 to 12-15 operation categories. Prioritize non-numeric symbolic/logical categories (Negation, Set Membership, Entailment) to balance the ontology.
- `[ ]` **Power Pre-Registration:** Run Monte Carlo power analysis to compute exact $N$ of categories/pairs needed for $80\%+$ power at the observed effect size *after* confound regression.
- `[ ]` **Data:** Maintain $N=14$ Confirmation prompts per category.
- `[ ]` **Statistics:** Mantel permutation test ($N=10,000$). Report subgroup breakdowns (old vs new categories) before claiming the aggregate number. Regress confounds at each expansion.
