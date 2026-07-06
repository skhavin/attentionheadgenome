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

*Prior Context (Finding 4):* Paper I/Phase II identified a depth-based split between Early Induction (prefix-matching, lower V/Q) and Late Induction (payload-copy, higher V/Q) heads. Phase III directly tests whether this structural depth split is actually a symptom of a temporal execution split (Prefill vs. Decode), moving beyond static architectural observation into runtime mechanics.

This category unifies systems-level execution modes (Prefill vs. Decode) with our mechanistic component taxonomy (Retrieval Heads, MLPs, Induction Heads). We treat the transformer as a dynamic factory that switches routing protocols based on its execution mode.

### The Mechanistic Mapping (Theory)
1. **The Parallel Spatial Engine (Prefill):** 
   * *Routing:* Information flows spatially across tokens. Attention heads dominate early layers to gather context. **Early Induction** heads perform prefix-matching here.
   * *Computation:* MLPs in the middle layers act as the logic factory. They read spatial vectors and physically rotate/forge them into complex semantic concepts (e.g., the "Concept of 7"). This forged concept is written into the KV Cache.
2. **The Autoregressive Memory Engine (Decode):**
   * *Routing:* Spatial interaction is broken. The model relies on temporal memory retrieval. **Late Induction** and Counting Heads dominate late layers, querying the static KV Cache to pull the forged concept to the final token. 
   * *RoPE Dependency Risk Pre-registration:* This "static cache" assumption holds elegantly for relative positional encodings (RoPE architectures like Llama/Qwen). However, for Absolute Positional Encodings (GPT-2), the temporal pipeline may fracture, mirroring our KV-eviction findings in Paper I. If GPT-2 fails these tests while Qwen passes, it will be explicitly reported as an architectural fracture, not a universal law.
   * *Computation:* MLPs act as context-sustainers, stabilizing the vector down the residual stream to project the next token.

### Experiment 0: The Temporal Handoff (Retrieval vs. Induction)
* **Hypothesis:** Retrieval heads (and Early Induction heads) fire predominantly during the Prefill phase, while Late Induction heads fire predominantly during the Decode phase.
* **Probe:** Measure the activation magnitudes of identified head populations strictly separated by execution phase. 
* **Sample Size Pre-registration:** Due to the rarity of Retrieval heads (1.5%, finding single-digit counts per architecture), we will pool data across architectures to ensure statistical power. We will not report fragile per-architecture N sizes as equally powered.

### Experiment 1: Information Weight Shift (Attention vs. MLP Norms)
* **Hypothesis:** The magnitude (L2 norm) of residual stream updates ($\Delta x$) is Attention-heavy early and MLP-heavy late during Prefill, but this symmetry breaks during Decode.
* **Probe:** Extract the exact $\Delta x$ norm from Attention sublayers vs MLP sublayers at every layer $L$, split cleanly between Prefill and Decode passes.
* **Note on Frobenius Axis:** This measurement axis (comparing sublayer-to-sublayer residual norms) is conceptually distinct from the within-attention V/Q Frobenius ratio analyzed in Finding 2. It will be flagged explicitly to prevent reviewers from conflating the two statistics.

### Experiment 2: Decode Symmetry Breaking (The Shift to Memory)
* **Hypothesis:** Because the KV cache handles spatial routing in Decode, Decode phases trigger deep MLP logic rotations much earlier in the network to stabilize the newly generated token against historical context.
* **Falsification:** If the Attention/MLP ratio across layers looks identical in Prefill and Decode, the Execution Duality hypothesis is falsified.

### Experiment 3: The Cross-Phase Intervention (The Ultimate Causal Test)
* **Hypothesis:** MLPs forge concepts in Prefill, and Decode strictly retrieves them. We can force a behavior change by swapping state across time modes.
* **Probe Task Spec (Crisp Binary Falsification):** To match the rigor of the NIAH ablation task in Paper I, the causal patch must use a strictly defined task template: `Question: What is 4 plus 3? Answer: The sum is`. The success metric is binary: does the argmax logit shift cleanly from an incorrect sum to `7`? The patched heads will be strictly isolated to the specific L16 Counting Heads identified in Phase II.
* **Causal Patch:** Ablate the Counting heads during Prefill (corrupting the KV cache). Surgically extract a healthy "Concept of 7" MLP vector from Layer 12 of a normal Prefill run, and patch it directly into Layer 12 of the broken Decode run.
* **Failure Structure:** If the model instantly switches its prediction to the correct answer (`7`), we have proven the strict handoff mechanism. If it fails, our mapping of the spatial-to-temporal pipeline is wrong.
