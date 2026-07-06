# HeadGenome III: Computational Development of Layers
**Core Question:** Why do transformers have layers, and how does a thought evolve from Token 1 to Token N?

This phase treats the **entire layer** as the fundamental unit of computation. We enforce a strict falsification-first methodology, pre-registering null baselines and explicit failure criteria for every experiment to avoid structural confounds.

## Relationship to Paper I

**Entropy collapse formula (Paper I):** The entropy-based head taxonomy (Sink, Local, Retrieval, Induction) is a *static structural* classifier measured on a single forward pass. It is phase-agnostic and is NOT affected by Experiment 0's falsification. The taxonomy remains valid as the population-identifier feeding into Phase III.

**Circuit 2 (Counting):** The causal efficacy of Counting Heads was established via mechanically zeroing head outputs and measuring downstream logit shift (Wilcoxon p≈0.0000, N=50). This was not based on output norms and is completely unaffected by Experiment 0's falsification.

## Proposed Directory Structure: `headgenome3_layers/`
```text
headgenome3_layers/
├── 01_execution_duality/       # Prefill vs. Decode execution pipeline mapping (PRIORITY 1)
├── 02_residual_evolution/      # Information birth/death via Logit Lens
└── 03_universal_stages/        # Exploratory observation of layer phases
```

## Category A: The Transformer Execution Duality (ICLR Flagship Thesis) ⭐⭐⭐⭐⭐

*Prior Context (Finding 4):* Paper I identified a depth-based split between Early Induction (prefix-matching, lower V/Q) and Late Induction (payload-copy, higher V/Q) heads. Phase III tests whether this structural depth split is a symptom of a **temporal** execution split (Prefill vs. Decode).

---

## Experiment 0: The Temporal Handoff — THE HARD GATE (Revised: MAD-Based)

### Prior Falsification
The original norm-based approach (v1 and v2) was falsified:
* Induction Decode/Prefill norm ratio: **1.27x** — identical to Local/Sink control (1.27x).
* Mann-Whitney p = 0.68 — Induction indistinguishable from control heads.
* Root cause: Output L2 norm is an architecture-wide artifact of single-token vs multi-token softmax averaging. It is not a signal of functional specialization.

### Revised Metric: Normalized Mean Attention Distance (nMAD)

**Precise Definition:**
* At each forward pass, identify the *final active token index* `t = last_idx`. In Prefill, `t = seq_len - 1` (last prompt token). In Decode, `t = prompt_len + num_generated_so_far - 1` (the single newly-generated token's position).
* Compute raw attention distance: $\text{rawMAD}_h = \sum_j \alpha_{h,t,j} \cdot (t - j)$, where $\alpha_{h,t,j}$ is the attention weight from token $t$ to position $j$.
* **Normalize by current sequence length:** $\text{nMAD}_h = \text{rawMAD}_h / t$. This maps nMAD to $[0, 1]$ regardless of absolute context length, making Prefill and Decode values directly comparable.
* A head with nMAD near 1.0 attends exclusively to position 0 (Sink). A head near 0.0 attends to the immediately adjacent token (Local). Induction/Retrieval heads attending to distant content have intermediate-to-high nMAD.

**Why this escapes the norm artifact:** nMAD measures *where* a head looks, normalized to be length-invariant. If Induction heads truly shift from co-present spatial routing (Prefill, where context is nearby) to deep history retrieval (Decode, where they must reach far back into the frozen KV prefix), their nMAD will increase. Local/Sink heads will not show this shift.

**Control:** Local/Sink heads should show **no significant nMAD shift** between Prefill and Decode.

### Pre-Registered Pass Criteria

All four must pass. There is no partial pass. If criteria 1–3 pass and NIAH (criterion 4) fails, the outcome is classified as "task-specific, requires cross-task extension" — the result is published as arithmetic-domain-only and **Experiments 1–3 must be restricted to the arithmetic domain** until NIAH replication succeeds.

1. **Induction Decode nMAD / Prefill nMAD > 1.5** (per-prompt mean, N=50)
2. **Wilcoxon signed-rank** (Decode nMAD > Prefill nMAD per prompt): $p < 0.05$
3. **Mann-Whitney U** (Induction nMAD shift > Control nMAD shift): $p < 0.05$
4. **NIAH cross-task validation:** Induction nMAD shift > 1.5 on N=50 NIAH prompts.

---

## Experiment 1: Information Weight Shift (Attention vs. MLP Norms)

**May only begin after Experiment 0 passes.**

* **Hypothesis:** Residual stream updates ($\Delta x$) are Attention-heavy early and MLP-heavy late during Prefill, but this symmetry breaks during Decode.
* **Measurement:** $\Delta x_{attn}^{(l)} = \|x_{after\_attn}^{(l)} - x_{before\_attn}^{(l)}\|_2$ and $\Delta x_{mlp}^{(l)} = \|x_{after\_mlp}^{(l)} - x_{before\_mlp}^{(l)}\|_2$ at each layer $l$, for N=50 prompts.
* **Phase Separation:** Computed separately for Prefill pass and Decode step.
* **Statistical Test:** **Paired Wilcoxon signed-rank** on per-layer (Attn $\Delta x$ / MLP $\Delta x$) ratio between Prefill and Decode. Layers are paired (same layer, same prompt, two phases). Mann-Whitney is NOT used here because layers are not independent samples.
* **Falsification:** If the Wilcoxon test on the Prefill-vs-Decode ratio profile finds $p \geq 0.05$, the Execution Duality hypothesis is falsified at the layer level.
* **Frobenius Disambiguation:** This $\Delta x$ measurement is conceptually distinct from the within-attention V/Q Frobenius ratio in Paper I Finding 2. It will be labeled explicitly in any writeup.

## Experiment 2: Decode Symmetry Breaking

* **Hypothesis:** The crossover layer $L^*$ (where MLP updates dominate Attention updates) occurs significantly earlier in the layer stack during Decode than during Prefill.
* **Measurement:** For each prompt, find $L^*_{Prefill}$ and $L^*_{Decode}$ as the first layer where $\Delta x_{mlp}^{(l)} > \Delta x_{attn}^{(l)}$.
* **Statistical Test:** Paired Wilcoxon on per-prompt $(L^*_{Decode} - L^*_{Prefill})$ — must be significantly negative ($p < 0.05$).

## Experiment 3: The Cross-Phase Intervention (The Ultimate Causal Test)

* **Hypothesis:** MLPs forge concepts in Prefill, and Decode strictly retrieves them.
* **Probe Task Spec:**
  * Template: `Question: What is X plus Y? Answer: The sum is`
  * N: 50 distinct `(X, Y)` pairs, single-digit sums 2–9, no pair repeated.
  * Success Metric: `argmax(logits)` shifts to the target sum.
  * **Pre-registered Pass Threshold:** Success rate $\geq 50\%$.
* **Causal Patch:** Ablate L16 Counting Heads (from `counting_heads_qwen-0.5b.json`) during Prefill. Patch a healthy Layer-12 MLP hidden state from a normal Prefill into Layer-12 of the broken Decode.
* **Null Control:** Same patch applied at Layer-5 (depth-matched non-critical). McNemar's test comparing Layer-12 vs Layer-5 success rates, $p < 0.05$ required.
* **Failure Structure:** If $\geq 50\%$ threshold is not met, our Prefill-forge/Decode-retrieve mapping is wrong for this task and must be revised.
