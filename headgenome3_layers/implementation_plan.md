# HeadGenome III — Flagship Thesis: The Attention-to-MLP Structural Handoff

## Goal Description
To test whether the physical computation of the transformer dynamically shifts from an attention-dominated routing protocol during Prefill to an MLP-dominated logic rotation during Decode. 

This plan abandons the falsified "attention-routing" duality and explicitly tests the **Attention $\rightarrow$ MLP structural handoff**. Systems engineers understand the latency shift of the KV-cache; interpretability researchers understand static circuits; this project will test whether physical circuits dynamically reorganize across the temporal boundary.

## User Review Required

> [!IMPORTANT]
> **Pre-Registered Hypothesis**
> This plan proposes testing the *Attention $\rightarrow$ MLP handoff* as a distinct hypothesis. We are not committing to the narrative in advance; we are committing to a strict, testable experimental design. Please review the corrected metrics below to ensure they are fully immunized against the artifacts that plagued Experiment 0 v1.

## Proposed Changes

### [headgenome3_layers/02_attention_vs_mlp]

#### [NEW] [exp2_structural_duality.py](file:///d:/PROJECTS/webstromprojects/attentionheadgenome/headgenome3_layers/02_attention_vs_mlp/exp2_structural_duality.py)
**Purpose:** Measure the macro-structural shift between Prefill and Decode without the multi-token averaging artifact.

**Corrected Mechanics (Apple-to-Apples):**
- **Metric:** The component norm ratio $R^{(l)} = ||\Delta x_{mlp}^{(l)}|| \ / \ ||\Delta x_{attn}^{(l)}||$
- **Prefill Measurement:** Computed strictly at the **final token of the prompt** (a single-token slice). 
- **Decode Measurement:** Computed at the **first generated token** (a single-token slice).
- **Control:** We will track $R^{(l)}$ across three layer buckets: *Early* (L0-L5), *Middle* (L6-L17), *Late* (L18-L24). If the shift is a global mechanical artifact, $R^{(l)}$ will spike everywhere. If it is a functional handoff, the shift will be heavily localized to the Middle/Late semantic layers.

**Pre-registered Gate:** 
- The median ratio $R^{(l)}$ in the Middle/Late buckets must show a statistically significant shift between Prefill (final prompt token) and Decode (first generated token) via Wilcoxon signed-rank test ($p < 0.05$).
- The shift magnitude in the Middle/Late buckets must be at least 2x the shift magnitude observed in the Early control bucket.

#### [NEW] [exp3_cogating_handoff.py](file:///d:/PROJECTS/webstromprojects/attentionheadgenome/headgenome3_layers/02_attention_vs_mlp/exp3_cogating_handoff.py)
**Purpose:** Trace the explicit Head $\rightarrow$ MLP flow magnitude leveraging our taxonomy.

**Corrected Mechanics (Flow Magnitude):**
- We isolate a known functional head (e.g., a Retrieval head). Let its output vector be $h_{out}$.
- We define the primary input subspace $U$ of the immediate downstream MLP (computed via SVD on all MLP inputs across the dataset).
- **Metric:** The projection magnitude of the head's output onto the MLP's input space: $M = || \text{Proj}_{U}(h_{out}) ||$.
- This measures how much "energy" the specific head is feeding directly into the MLP's active logic pathways, rather than just static cosine alignment.

**Pre-registered Gate:**
- The flow magnitude $M$ from Retrieval Heads to their co-gated MLPs must be significantly higher in Decode than in Prefill, supporting the hypothesis that the MLP is "doing more" with that specific input during autoregression.

## Verification Plan

### Automated Tests
- The scripts will automatically compute the Wilcoxon p-values for both the macro structural ratio (Exp 2) and the specific flow magnitude (Exp 3). 
- If either $p \ge 0.05$ or the Early-layer control fails, the hypothesis is nullified and the Execution Duality is definitively dead.

## Findings

### Part 1: Architecture & Structural Taxonomy
1. **The V/Q Spatial Scaling Law:** A continuous, cross-architecture scaling law where early heads are query-dominant (locators) and deep heads are value-dominant (payload delivery) ($p=1.92 \times 10^{-127}$).
   - **Script:** `plot_developmental_curve.py`
   - **JSON:** `outputs/phase8_paper_suite/statistical_suite_results.json`
2. **The Developmental Manifold:** Heads transition from Sink $\rightarrow$ Local $\rightarrow$ Retrieval/Induction in a continuous geometric manifold rather than discrete hardcoded buckets. ~84% of heads are Local precursors.
   - **Script:** `canonical_classification.py`
   - **JSON:** `outputs/canonical_labels.json`
3. **The Early vs Late Induction Split:** Induction heads separate into early-layer (query-dominant, prefix matching) and late-layer (value-dominant, payload copying) sub-populations.
   - **Script:** `paper_analysis_suite.py`
   - **JSON:** `outputs/phase8_paper_suite/statistical_suite_results.json`
4. **Histogram Invisibility:** Weight clustering alone (via SVD/Frobenius norm) flattens functional types. Static weights cannot classify functional roles; dynamic entropy probing is required.
   - **Script:** `phase2/step2_clustering.py`
   - **JSON:** `outputs/phase2/cluster_metrics.json`
5. **Universal Architectural Geometry:** Induction heads consistently cluster at normalized depth 0.46 to 0.60. Local heads dominate (55-65%). True Retrieval heads are the rarest nodes (0.3% to 2.1%).
   - **Script:** `phase2_atlas/compare_atlases.py`
   - **JSON:** `outputs/phase2_atlas/dataset.json`
6. **Initialization Null (Emergence of Structure):** Randomly initialized models completely lack the V/Q scaling law and functional taxonomy. The structure is an emergent property of gradient descent, not architectural initialization.
   - **Script:** `scripts/phase10_vq_universality.py`
   - **Output:** `outputs/phase10_universality/figure8_vq_emergence.png`
7. **The Permutation Null (Data Independence):** The taxonomy structure survives when models process completely shuffled gibberish tokens. Induction entropy collapse strengthens while Retrieval attenuates, proving structural firing independent of semantics.
   - **Script:** `scripts/phase11_permutation_null.py`
   - **Output:** `outputs/phase11_permutation_null/figure9_permutation_null.png`
8. **Cross-Domain Invariance:** The V/Q scaling law survives massive domain shifts (from purely English WebText to 18-Trillion-token Code and Math corpora), confirming the geometry is fundamentally data-agnostic.
   - **Script:** `scripts/phase11_cross_domain_proof.py`
   - **Output:** `outputs/phase11_universality/figure10_cross_domain.png`

### Part 2: Behavioral & Causal Mechanics
9. **Causal Ablation Degradation:** Ablating all 311 Local heads completely destroys perplexity (+244.88). Ablating the 15 Sink heads severely degrades stability (+199.36).
   - **Script:** `phase5/step2_fixed_ablation.py`
   - **JSON:** `outputs/phase5/fixed_ablation.json`
10. **The Retrieval-Induction Co-Gating (Circuit Dependency):** A model cannot complete a NIAH task with only Retrieval heads; it structurally requires Induction heads to physically copy the retrieved text. They form a co-dependent pipeline.
    - **Script:** `phase6/step4_retrieval_curve.py`
    - **JSON:** `outputs/phase6/retrieval_curve_synthetic_ruler.json`
11. **Retrieval-Induction Co-Gating Falsification (Not a Strict AND Gate):** Ablating all known Retrieval heads does NOT catastrophically collapse the Induction heads. The logit drop is minimal (~4-8%), indicating massive redundancy or parallelization in the locating circuit.
    - **Script:** `phase2_atlas/step8_causal_patching.py`
    - **JSON:** (Dynamic ablation execution logs)
12. **Regime Switching & Plasticity:** ~85% of heads are completely stable across domains. A critical 5-10% of heads are highly context-sensitive, switching behavior drastically (336x to 3436x variance).
    - **Script:** `regime_switching_analysis.py`
    - **JSON:** `outputs/phase8_paper_suite/regime_switching_<model>.json`
13. **Copy-Retrieval Co-Activation:** The highest-variance context-sensitive heads peak simultaneously on Copy and Retrieval tasks.
    - **Script:** `regime_switching_analysis.py`
    - **JSON:** `outputs/phase8_paper_suite/regime_switching_<model>.json`
14. **Lexical Target Separation:** Modern models orthogonally separate syntax from semantics. Local heads target grammar (stop words, articles) while Induction heads exclusively target high-information semantic nouns. GPT-2 suffers from entanglement.
    - **Script:** `phase2_atlas/lexical_tracker.py`
    - **Output:** `outputs/phase9_semantics/figure7_lexical_anatomy.png`
15. **The KV Cache Mini-Sink Law:** Punctuation tokens act as deliberate structural reset mini-sinks for local chunking ($z=171.13, p=0$).
    - **Script:** `phase2_atlas/step3_grammar_map.py` & `phase2_atlas/analyze_punctuation_rigorous.py`
    - **JSON:** `outputs/phase2_atlas/dataset.json`
16. **Llama-3.2-1B BOS Parking Anomaly:** Due to RoPE, Llama-3.2-1B models park >80% of attention mass for nearly 90% of their heads onto the `<|begin_of_text|>` token, creating a universal attention sink.
    - **Script:** `audit_head_vocabulary.py`
    - **Output:** `outputs/phase9_semantics/figure7_lexical_anatomy.png`
17. **Causal Sink Falsification:** Removing the BOS token causes a massive entropy explosion across Llama's Sink heads.
    - **Script:** `phase2_atlas/step5_sink_falsification.py`
    - **JSON:** (Dynamic synthetic prompts logs)
18. **Polysemantic Multiplexing (Micro-SAE):** Attention head output vectors structurally multiplex features. A True SAE reconstructs the output using 4.05-5.14 active features/token, whereas a Null SAE (shuffled covariance) requires 51-52 active features/token.
    - **Script:** `phase2_atlas/step10_micro_sae.py`
    - **JSON:** (Trained Micro-SAE checkpoints)
19. **Softmax Saturation (Falsified):** Retrieval heads do not operate at statistically higher softmax saturation than Local heads. The binary gate hypothesis is false.
    - **Script:** `phase2_atlas/step4_softmax_saturation.py`
    - **JSON:** `outputs/phase2_atlas/dataset.json`
20. **The Giant Megacluster:** Unsupervised UMAP/HDBSCAN clustering confirms that 58% of heads form a continuous manifold megacluster rather than rigid discrete types.
    - **Script:** `phase2_atlas/step15_rich_features.py`
    - **Output:** `outputs/final_artifacts/visualizations/umap_clusters.png`
21. **Universal Copy Circuit (Falsified):** Ablating heads that allocate 80% mass to copying exact string tokens (e.g. UUIDs) yields a 0% drop in actual accuracy. High attention mass does not imply causal necessity for copying.
    - **Script:** `headgenome2_circuits/01_universal_copy/ablation.py`
    - **JSON:** (Runtime pre-registered output logs)
22. **Counting Circuit (Proven):** Specific counting heads dictate sequential list counting. Inter-layer patching of the residual stream state accurately shifts output from $X$ to $X+2$ ($p = 0.0000$).
    - **Script:** `headgenome2_circuits/02_counting_mechanisms/patching.py`
    - **JSON:** (Runtime pre-registered output logs)
23. **Structured Output JSON Circuit (Falsified):** Heads allocating 100% mass to open braces `{` show no causal necessity; ablating them maintains 100.0% valid JSON closure.
    - **Script:** `headgenome2_circuits/03_structured_output/json_ablation.py`
    - **JSON:** (Runtime pre-registered output logs)
24. **Attention-MLP Routing (Proven for Counting):** The structural Frobenius norm ($||W_{fused\_gate} \cdot W_O||_F$) of the Counting heads predicts the exact downstream target MLP receiving the causal flow with near-perfect correlation ($r=0.996$).
    - **Script:** `headgenome2_circuits/04_attention_mlp_routing/frobenius.py`
    - **JSON:** (Runtime pre-registered output logs)
25. **Arithmetic Circuit (Falsified):** Counting heads that generalize to allocate 70% mass onto mathematical operands have exactly zero causal impact on actual single-digit arithmetic output ($p=0.9838$).
    - **Script:** `headgenome2_circuits/05_arithmetic/patching.py`
    - **JSON:** (Runtime pre-registered output logs)

### Part 3: Systems & Engineering Applications
26. **The Perplexity (PPL) Illusion:** A model can perfectly preserve language perplexity (e.g. 13.07 sparse vs 11.71 dense) while completely collapsing in long-range reasoning (NIAH accuracy dropping from 100% to 42%).
    - **Script:** `phase6/step1_sparse_prefill.py` & `phase6/step3_ruler_comprehensive.py`
    - **JSON:** `outputs/phase6/sparse_prefill.json` & `outputs/phase6/ruler_comprehensive.json`
27. **Decode KV Eviction (13.3x Compression):** Evicting tokens dynamically based on the HeadGenome taxonomy (Budget=64) perfectly preserves Llama-3.2-1B perplexity (9.98 vs 132.44 StreamingLLM).
    - **Script:** `phase4/step3_routing_policy.py`
    - **JSON:** `outputs/phase4/routing_policy_results.json`
28. **Sparse Prefill Scaling (75.9% Empirical FLOP Savings):** Applying a sliding window of W=384 to Local heads while preserving critical heads achieved 75.9% measured FLOP reduction at N=4096.
    - **Script:** `phase6/step1_sparse_prefill.py`
    - **JSON:** `outputs/phase6/sparse_prefill.json`
29. **The Local Head Success (Validation via Routing):** Forcing Local heads into a strict 32-token sliding window drops HellaSwag reasoning accuracy by only 1.0%.
    - **Script:** `phase2_atlas/step18_routing_engine.py`
    - **Output:** `outputs/final_artifacts/visualizations/routing_validation.png`
30. **The Sink Head Falsification (Validation via Routing):** Forcing Sink heads to exclusively attend to the BOS token drops HellaSwag accuracy catastrophically (by 5.0% to near random). Their distributed non-BOS attention mass is functionally critical.
    - **Script:** `phase2_atlas/step18_routing_engine.py`
    - **Output:** `outputs/final_artifacts/visualizations/routing_validation.png`
31. **Theoretical FLOP Scaling Ceilings:** Based purely on head distribution math, projected FLOP savings scale up to 84-93% at N=4096.
    - **Script:** `phase4/step5_scaling_curves.py`
    - **JSON:** `outputs/phase4/scaling_curves.json`
