# HeadGenome III: Computational Development of Layers

## Quick Reference

| Category | Status | Scope |
|---|---|---|
| **A: Execution Duality (Prefill vs. Decode)** | ❌ TOMBSTONED | Falsified (nMAD gate failed) |
| **B: Residual Evolution (Logit Lens)** | 🟡 ACTIVE | Experiment 1 — pre-registered |
| **C: Universal Stages** | ⏳ PENDING | After Category B lands |

---

## Category A: The Transformer Execution Duality — TOMBSTONED

### Full Arc (Preserved for Paper/Appendix)

The "Execution Duality" hypothesis was that Prefill and Decode represent functionally distinct computational regimes, operationalized as: Induction/Retrieval heads should allocate attention dramatically differently depending on which phase is running.

**Measurement v1 (norm-based, May 2026):**
Used head output L2 norm ($\|W_O x_h\|_2$) as a proxy for phase activation. Found a 3.75× Decode/Prefill ratio for Induction heads.

**Artifact identification:**
The 3.75× result was a measurement bug: with `max_new_tokens=1`, HuggingFace's `generate()` places the full prompt in the "Decode" bucket when the hook fires on the single generation step. Single-token softmax normalization differs from multi-token row-averaged Prefill norms by construction, regardless of what the heads are doing. The 3.75× was a normalization artifact, not a functional signal.

**Redesigned metric (nMAD):**
$\text{nMAD}_h = \frac{\sum_j \alpha_{h,t,j} \cdot (t - j)}{t}$

where $t = k\_len - 1$ (absolute current token position), $\alpha$ is drawn from the last query row only, and the division by $t$ maps the result to $[0,1]$ regardless of context length. This metric is mathematically invariant to phase-induced length differences and directly comparable across Prefill and Decode passes.

**Results (pre-registered thresholds, N=50, qwen-0.5b):**

| Condition | Prefill nMAD | Decode nMAD | Ratio | Wilcoxon p | Gate |
|---|---|---|---|---|---|
| Induction / Arithmetic | 0.4454 | 0.4698 | 1.05× | <0.0001 | FAIL (need >1.5) |
| Control (Local/Sink) | 0.5389 | 0.5445 | 1.01× | <0.0001 | — |
| Induction / NIAH | 0.4660 | 0.4802 | 1.03× | <0.0001 | FAIL (need >1.5) |
| Mann-Whitney (Ind > Ctl) | — | — | — | <0.0001 | PASS |

The direction was correct (Induction shifts slightly more than Control, $p < 0.001$ Mann-Whitney). The magnitude is negligible (1.05× vs threshold 1.5×). The hypothesis fails even after fixing the measurement.

**What is and is not falsified:**

> **Dead:** The attention-routing half of the duality. Induction/Retrieval heads do not redistribute their attention mass across the Prefill/Decode boundary in any functionally meaningful way. The nMAD distribution in Prefill (~0.45) is near-identical to Decode (~0.47). Prefill is not just "forging" — it is heavily retrieving throughout.

> **Not tested:** The MLP half. Whether MLP residual-stream update magnitudes ($\Delta x_{mlp}^{(l)}$) differ by phase was gated on Experiment 0 passing. That test was never run. The MLP story ("MLPs forge in Prefill, Decode context-sustains") is unresolved, not falsified. If the MLP side is to be fully retired, Experiment 1 (from the original plan) must be run as an **ungated standalone** test.

> **Open question:** Systems-level sparsification methods (sparse Prefill, decode-only KV eviction) do produce very different efficiency gains in the two phases, which seems to conflict with the null result here. The reconciliation: those methods exploit causal masking + cache mechanics (where Prefill processes the entire prompt in a single dense forward pass and Decode processes one token at a time), not head-level attention specialization. Our null result says **heads don't care which phase they're in**; the systems-level divergence is a property of the KV cache mechanics, not of learned head behavior. This is worth a sentence in the paper — it explains why the systems literature treats Prefill and Decode differently without implying heads are functionally specialized by phase.

**Methodological contribution:**
This arc — plausible signal → identified measurement artifact → redesigned invariant metric → pre-registered gate → clean failure — is a worked example of the kind of falsification-first methodology the interpretability literature needs. The control group (Local/Sink heads showing 1.01× ratio alongside Induction's 1.05×) was the decisive evidence. Reviewers who have been burned by unfalsifiable mechanistic claims should find this valuable.

---

## Category B: Residual Evolution (Logit Lens) — ACTIVE

### Background and Motivation
Following the Category A falsification, we know heads do not bifurcate by phase. The next natural question is: how does the *residual stream* evolve across depth? At what layer does the correct answer "emerge" as the top prediction? Is this emergence sudden (circuit-completion-style) or gradual (ensemble of weak votes)?

This is the closest thing to a hard, falsifiable binary-outcome experiment remaining. Category C ("Universal Stages") is explicitly exploratory and runs after Category B to use the Logit Lens result as an anchor.

### Experiment 1: Sudden vs. Gradual Emergence (THE GATE)

**Task continuity:** Reuse exactly the N=50 arithmetic prompts and N=50 NIAH prompts from Experiment 0. Same inputs → cross-referenceable against the head-level nMAD measurements already in hand.

**Metric:** Per-layer Logit Lens probability. At each layer $l$, pass the residual stream $x^{(l)}$ through the final LayerNorm and unembedding matrix to get a probability distribution over vocabulary. Extract the probability assigned to the *correct answer token* (e.g., the digit `7` for `4+3=`, or the UUID digits for NIAH).

$P^{(l)} = \text{softmax}(\text{LN}(x^{(l)}) W_U)[\text{target}]$

**Emergence layer $L^*$:** The first layer where the target token reaches **top-1 rank** (overtakes all competitors). Binary and unambiguous.

**Sudden emergence metric:**
Compute $\delta^{(l)} = P^{(l)} - P^{(l-1)}$ for each layer. Let $\delta_{max} = \max_l \delta^{(l)}$ and $\Delta_{total} = P^{(L)} - P^{(0)}$ (total probability gain from first to last layer). Define:

$$S = \frac{\delta_{max}}{\Delta_{total}}$$

$S > 0.40$ means a single layer accounts for >40% of the total probability gain. This is the pre-registered threshold for "sudden."

**Null control: Shuffled-Prompt Permutation Null.** Take the same N=50 prompts, randomly permute the token order within each prompt (destroying semantic content but preserving token distribution and sequence length). Run the same pipeline. Track the trajectory of the *same target token* (the original correct answer). This tests whether the sharp emergence jump is real or a structural artifact of how Logit Lens trajectories look under any input.

### Pre-Registered Pass Criteria (Locked — Do Not Modify After Running)

**PASS requires ALL four:**
1. **Coverage:** $L^*$ is defined (top-1 rank achieved before the final layer) for ≥80% of prompts.
2. **Sudden magnitude:** $S > 0.40$ — the single largest layer-to-layer jump accounts for >40% of total probability gain, averaged across prompts that have a defined $L^*$.
3. **Wilcoxon specificity:** $S_{\text{real}} > S_{\text{shuffled}}$ per-prompt, Wilcoxon signed-rank $p < 0.05$.
4. **Cross-task replication:** Criteria 1–3 hold independently on both arithmetic and NIAH prompt families.

**Partial-pass rule (pre-registered):**
- If 1–3 pass on arithmetic but not NIAH: "arithmetic-domain finding." Proceed to Category C on arithmetic only.
- If any of 1–3 fail on arithmetic: **hard stop**, reassess hypothesis.

**What "sudden" would mean:** The residual stream undergoes a discrete state transition at a specific layer — consistent with the "circuit-completion" model of transformer computation. Individual attention heads or MLP blocks at that layer are causal targets worth investigating.

**What "gradual" would mean:** The transformer operates as a smooth ensemble accumulator. No single layer is privileged; interpretability must be done at the ensemble level.

**Per-architecture plan:** Run independently on qwen-0.5b, qwen-1.5b, gpt2. Llama-1b and gemma-2b added if resources allow. Cross-architecture universality is a secondary analysis only after per-architecture results are in.

---

## Category C: Universal Stages (Exploratory) — PENDING

Not pre-registered. Runs after Category B. Uses the Logit Lens $L^*$ layers as anchors to identify whether early layers (1 to $L^*/2$) show qualitatively different residual stream content (syntax-heavy) vs. late layers ($L^*/2$ to $L$) (semantics-heavy). Explicitly exploratory — no hard gate, no claim stronger than "suggestive pattern."

---

## Visualizer Plan

**Goal:** A real-time Streamlit app where a user types a prompt and sees the full internal computation of the model: per-layer attention maps, per-layer Logit Lens probability trajectories, residual stream norms, and head-type annotations — all synchronized to the same forward pass.

**Interface:** Streamlit (not tkinter — runs in browser, shareable, supports interactive Plotly charts). Chat-style prompt input on the left; visualization panels on the right, each collapsible.

**Panels (in priority order):**
1. **Logit Lens Panel:** Line chart of $P^{(l)}[\text{top-5 tokens}]$ across all layers. Highlights $L^*$ with a vertical marker. Correct token (if known) shown in a distinct color.
2. **Attention Heatmap Panel:** Per-layer, per-head attention matrix (softmax weights) for the last token. Head-type label (Induction / Local / Sink / Unknown) shown as color-coded badge from Paper I's taxonomy.
3. **Residual Stream Norm Panel:** $\|x^{(l)}\|_2$ plotted per layer, split into pre-Attention, post-Attention, post-MLP contribution ($\Delta x_{attn}$ and $\Delta x_{mlp}$ stacked bar chart).
4. **nMAD Panel:** Per-head nMAD for Prefill pass, displayed as a heatmap (layer × head) with color encoding 0 (Local) → 1 (Far-reaching).
5. **Token Probability Table:** At the final layer, a ranked list of top-10 predicted tokens with log-probabilities.

**Script location:** `headgenome3_layers/visualizer/app.py`
**Dependencies:** streamlit, plotly, transformers, torch (already installed)
