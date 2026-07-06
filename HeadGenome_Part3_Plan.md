# HeadGenome III: Computational Development of Layers
**Core Question:** Why do transformers have layers, and how does a thought evolve from Token 1 to Token N?

This phase treats the **entire layer** as the fundamental unit of computation. We enforce a strict falsification-first methodology, pre-registering null baselines and explicit failure criteria for every experiment to avoid structural confounds.

## Proposed Directory Structure: `headgenome3_layers/`
```text
headgenome3_layers/
├── 01_execution_duality/       # Prefill vs. Decode execution pipeline mapping (PRIORITY 1)
├── 02_residual_evolution/      # Information birth/death via Logit Lens
└── 03_universal_stages/        # Exploratory observation of layer phases
```

## Category A: The Transformer Execution Duality (ICLR Flagship Thesis) ⭐⭐⭐⭐⭐

*Prior Context (Finding 4):* Paper I/Phase II identified a depth-based split between Early Induction (prefix-matching, lower V/Q) and Late Induction (payload-copy, higher V/Q) heads. Phase III directly tests whether this structural depth split is actually a symptom of a **temporal** execution split (Prefill vs. Decode), moving beyond static architectural observation into runtime mechanics.

This category unifies systems-level execution modes (Prefill vs. Decode) with our mechanistic component taxonomy (Retrieval Heads, MLPs, Induction Heads). We treat the transformer as a dynamic factory that switches routing protocols based on its execution mode.

### The Mechanistic Mapping (Theory)
1. **The Parallel Spatial Engine (Prefill):** 
   * *Routing:* Information flows spatially across tokens. Attention heads dominate early layers to gather context. **Early Induction** heads perform prefix-matching here.
   * *Computation:* MLPs in the middle layers act as the logic factory. They read spatial vectors and physically rotate/forge them into complex semantic concepts (e.g., the "Concept of 7"). This forged concept is written into the KV Cache.
2. **The Autoregressive Memory Engine (Decode):**
   * *Routing:* Spatial interaction is broken. The model relies on temporal memory retrieval. **Late Induction** and Counting Heads dominate late layers, querying the static KV Cache to pull the forged concept to the final token. 
   * *RoPE Dependency Risk Pre-registration:* This "static cache" assumption holds elegantly for relative positional encodings (RoPE architectures like Llama/Qwen). However, for Absolute Positional Encodings (GPT-2), the temporal pipeline may fracture, mirroring our KV-eviction findings in Paper I. If GPT-2 fails these tests while Qwen passes, it will be explicitly reported as an architectural fracture, not a universal law.
   * *Computation:* MLPs act as context-sustainers, stabilizing the vector down the residual stream to project the next token.

## Experiment 0: The Temporal Handoff (THE HARD GATE) — Revised After Norm Falsification

### What Was Falsified (v1 + v2 Results)
The initial Experiment 0 measured **head output L2 norm** ($\|W_O \cdot x_h\|_2$) across Prefill and Decode. This was falsified:
* Induction Decode/Prefill norm ratio: **1.27x** — identical to the Local/Sink control group (1.27x).
* Mann-Whitney p = 0.68 — Induction heads statistically indistinguishable from control.
* NIAH cross-task ratio: **0.97x** — no Decode dominance on the correct Retrieval task.
* Root cause: Output norm is an architecture-wide artifact of single-token vs multi-token averaging, not a signal of functional specialization.

### The Correct Signal: Mean Attention Distance (MAD)
Output *magnitude* does not reveal *where* a head looks. The correct signal is the **distributional shift in attention mass** across the KV sequence.

**Metric (MAD):** For each head $h$ at each forward pass, compute:
$$\text{MAD}_h = \sum_j \alpha_{h, \text{last}, j} \cdot (\text{last\_idx} - j)$$
where $\alpha_{h, \text{last}, j}$ is the attention weight from the final active token to position $j$. High MAD = head looks far back into history (Retrieval/Induction behavior). Low MAD = head attends locally (Local/Sink).

**Revised Hypothesis:** Induction heads show a significantly **higher MAD during Decode than during Prefill** — they reach further back into the frozen KV cache during Decode, whereas in Prefill the same tokens are spatially adjacent.

**Control:** Local/Sink heads should show **no significant MAD shift** between Prefill and Decode.

### Pre-Registered Pass Criteria (Locked Before Re-Running)
1. Induction Decode MAD / Prefill MAD **> 1.5** (per-prompt mean, N=50)
2. **Wilcoxon signed-rank** (Decode MAD > Prefill MAD): $p < 0.05$
3. **Mann-Whitney U** (Induction MAD shift > Control MAD shift): $p < 0.05$
4. **NIAH cross-task validation:** MAD shift > 1.5 on NIAH prompts as well.

**All four criteria must pass before Experiments 1–3 may begin.**


## Experiment 1: Information Weight Shift (Attention vs. MLP Norms)

**This experiment may only begin after Experiment 0 passes all four criteria above.**

* **Hypothesis:** The magnitude (L2 norm) of residual stream updates ($\Delta x$) is Attention-heavy early and MLP-heavy late during Prefill, but this symmetry breaks during Decode.
* **Measurement:** $\Delta x_{attn}^{(l)} = \|x_{after\_attn}^{(l)} - x_{before\_attn}^{(l)}\|_2$ and $\Delta x_{mlp}^{(l)} = \|x_{after\_mlp}^{(l)} - x_{before\_mlp}^{(l)}\|_2$ at each layer $l$.
* **Phase Separation:** Report these norms twice: once for the Prefill pass and once for the first Decode step.
* **Control:** Compute the per-layer Attention/MLP ratio for Prefill and Decode separately.
* **Falsification:** The Execution Duality hypothesis is falsified if the ratio profiles are not statistically distinguishable between phases (Mann-Whitney across layers, $p < 0.05$ required).
* **Frobenius Disambiguation:** This $\Delta x$ measurement is conceptually and mathematically distinct from the within-attention V/Q Frobenius ratio in Finding 2 and will be labeled explicitly to prevent reviewer conflation.

## Experiment 2: Decode Symmetry Breaking (The Shift to Memory)

* **Hypothesis:** Decode phases trigger deep MLP logic rotations significantly earlier in the layer stack than Prefill phases.
* **Measurement:** Find the "crossover layer" $L^*$ where $\Delta x_{mlp}^{(l)} > \Delta x_{attn}^{(l)}$ first occurs in each phase. 
* **Falsification:** If $L^*_{Decode}$ is not statistically earlier than $L^*_{Prefill}$ across multiple prompts (N=50, Wilcoxon), the symmetry-breaking hypothesis fails.

## Experiment 3: The Cross-Phase Intervention (The Ultimate Causal Test)

* **Hypothesis:** MLPs forge concepts in Prefill, and Decode strictly retrieves them. We can force a behavior change by swapping state across time modes.
* **Probe Task Spec (Pre-registered Binary Metric):**
  * **Template:** `Question: What is X plus Y? Answer: The sum is`
  * **N:** 50 distinct `(X, Y)` pairs, single-digit sums 2–9, no pair repeated.
  * **Success Metric:** On each prompt, `argmax(logits)` must shift from the incorrect baseline sum to the target patched sum. We report aggregate success rate.
  * **Pre-registered Pass Threshold:** Success rate $\geq 50\%$ (at least half the prompts show the predicted discrete shift). 
* **Causal Patch:** Ablate the L16 Counting Heads (from `counting_heads_qwen-0.5b.json`) during Prefill. Surgically patch a healthy Layer-12 MLP concept vector from a normal Prefill run into Layer-12 of the broken Decode run.
* **Null Control:** Apply the patch at a depth-matched non-critical layer (e.g., Layer-5 instead of Layer-12). Success rate must be significantly higher for Layer-12 than Layer-5 (McNemar's test, $p < 0.05$).
* **Failure Structure:** If success rate is not $\geq 50\%$ on the targeted patch, our Prefill-forge / Decode-retrieve mapping is wrong for this task and must be revised.
