# HeadGenome II: Algorithmic Circuits
**Core Question:** What functional algorithms are implemented inside transformer circuits?

This phase investigates exactly *how* multiple heads cooperate to execute specific computational motifs. Every experiment is designed with a strict falsification-first methodology: we define the null baseline, the causal intervention, and the exact failure criterion *before* the experiment.

## Proposed Directory Structure: `headgenome2_circuits/`
```text
headgenome2_circuits/
├── 01_universal_copy/          
├── 02_counting_mechanisms/     
├── 03_structured_output/       
├── 04_attention_mlp_routing/   
└── utils/                      
```

## Category A (Genuinely Open & High Impact)

### 1. The Universal Copy Circuit ⭐⭐⭐⭐⭐
* **Hypothesis:** Models possess a conserved mechanism for exact string copying (UUIDs, rare identifiers) that is distinct from standard semantic Induction.
* **Probe:** Feed synthetic UUIDs and calculate attention mass directed at previous occurrences.
* **Causal Intervention:** Ablate the identified "Copy" heads via `o_proj` zeroing.
* **Null Baseline / Control:** Ablate a randomly sampled set of non-copy heads that are strictly **depth-matched and entropy-matched**. (e.g. sample heads from the exact same layers with similar base attention concentration).
* **Falsification Criterion:** The hypothesis fails if the targeted ablation causes no statistically significant drop in exact-match string accuracy compared to the matched random-head ablation control.

### 2. The Counting Circuit ⭐⭐⭐⭐⭐
* **Hypothesis:** Transformers possess specific heads that act as accumulators, tracking quantities of delimiters or repeated items.
* **Probe:** Prompts requiring exact counting: "Item 1, Item 2, Item 3... What is the next item?"
* **Causal Intervention:** Inter-layer patching. Patch the residual stream state of a "Count=3" context into a "Count=5" context.
* **Null Baseline / Control:** Patch the state from a depth-matched *random* layer/head instead of the hypothesized "counting" head.
* **Pre-registered Success/Failure:** We pre-register a **Wilcoxon signed-rank test** ($N=50$, $\alpha=0.05$) comparing the shift in patched vs random-patched output distributions. Success requires a statistically significant positive rank shift toward $+2$. Failure is explicitly defined as $p > 0.05$ (effect indistinguishable from random patching).

### 3. Structured Output (JSON / Markdown) ⭐⭐⭐⭐⭐
* **Hypothesis:** Specific algorithmic circuits are solely responsible for maintaining nested state architectures (e.g., matching `{` with `}`).
* **Probe:** Feed the model deeply nested JSON blocks and prompt it to close them.
* **Causal Intervention:** Knock out the identified syntax/delimiter heads.
* **Falsification Criterion:** We define a strict quantitative metric: **Bracket-Matching Validity Rate** across $N=50$ parsed JSON structures. The hypothesis fails if ablating these specific heads does not drop the strict JSON parsing validity rate significantly more than ablating an equivalent set of depth-matched random heads ($p > 0.05$ via paired t-test).

### 4. Attention $\leftrightarrow$ MLP Cooperation ⭐⭐⭐⭐⭐
* **Hypothesis:** Attention acts as a router that writes specific queries into the residual stream explicitly to activate target MLP knowledge neurons.
* **Probe (The Frobenius Interaction Norm):** Mathematically compute the Frobenius Norm of $W_{out}^{(Attention)} \times W_{in}^{(MLP\_Next\_Layer)}$.
* **Depth-Control & Causal Validation:** We will mandate partial correlation against layer depth from the start. We will also validate causally: correlate the structural interaction norm against actual MLP activation shifts during patching.
* **Cross-Model Pre-registration:** Given previous history, we pre-register that results will be reported as a **mixed matrix**. A "universal success" requires $p < 0.05$ partial correlation on all 4 architectures. A mixed result (e.g., holds on Llama/Qwen but fails on GPT-2) will be reported explicitly as an architectural divergence, not a universal law.
