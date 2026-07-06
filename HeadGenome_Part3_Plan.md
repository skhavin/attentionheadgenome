# HeadGenome III: Computational Development of Layers
**Core Question:** Why do transformers have layers, and how does a thought evolve from Token 1 to Token N?

This phase treats the **entire layer** as the fundamental unit of computation. We enforce a strict falsification-first methodology, pre-registering null baselines and explicit failure criteria for every experiment to avoid structural confounds.

## Proposed Directory Structure: `headgenome3_layers/`
```text
headgenome3_layers/
├── 01_layer_bottlenecks/       # Layer-wise ablation and knock-out studies (PRIORITY 1)
├── 02_residual_evolution/      # Information birth/death via Logit Lens
├── 03_attention_vs_mlp/        # Frobenius norm analysis of layer updates
└── 04_universal_stages/        # Exploratory observation of layer phases
```

## Category A (Genuinely Open & High Impact)

### 1. Layer Bottlenecks & Inter-Layer Communication (Top Priority) ⭐⭐⭐⭐⭐
* **Hypothesis:** Removing a single layer will catastrophically destroy specific reasoning capabilities if the model relies on strict stage-gated development.
* **Probe:** Systematically zero out the residual stream updates for Layer $N$ across a benchmark suite (e.g., MMLU, HellaSwag).
* **Falsification/Failure Structure:** We pre-register **THREE** explicitly bounded outcomes across the layer knockout profile:
  1. **Catastrophic Isolation:** Specific task accuracy drops to chance (proving a strict computational bottleneck).
  2. **Graceful Uniform Degradation:** All tasks drop evenly and slightly (falsifying bottlenecks, proving continuous distributed refinement).
  3. **Uneven but Graceful Grading:** Specific tasks drop more than others, but none reach chance. We will not round this to "catastrophic". It will be explicitly classified as a **mixed-dependency architecture**.
* **Null Control:** Compare the ablation of a targeted "reasoning" layer against the ablation of a matched intermediate layer. 

### 2. Residual Stream Evolution (Information Birth & Death) ⭐⭐⭐⭐⭐
* **Hypothesis:** Semantic information undergoes a strict "birth" in the residual stream at a specific depth.
* **Probe (The Logit Lens):** Project the hidden state of the residual stream at every layer $L$ directly into the vocabulary space.
* **Null Baseline / Control:** The Logit Lens can show cosmetic "premature" answers. We cross-check any "birth layer" against a strict causal patching test: if we patch that layer's state into a corrupted run, does it actually change the final output?
* **Falsification Criterion:** The hypothesis fails if the "birth layer" identified by the Logit Lens does not have a causal impact on the final output when patched.

### 3. Attention vs. MLP Dominance (The Frobenius Flow) ⭐⭐⭐⭐⭐
* **Hypothesis:** Early layers are Attention-dominant, middle layers are MLP-dominant.
* **Probe:** Calculate the $Attention / MLP$ Frobenius Norm ratio at each layer.
* **Depth-Control & Falsification:** We mandate partial correlation against absolute layer depth from the start.
* **Cross-Model Pre-registration:** We explicitly pre-register that results will be reported as a **mixed matrix**. A "universal phase" requires partial correlation survival across all 4 architectures. Mixed results (e.g., holds in Qwen, fails in GPT-2) will be reported as architectural divergences.

### 4. Universal Layer Stages (Exploratory Only)
* **Pre-registration:** Because this is highly prone to overclaiming, this is explicitly downgraded to an *exploratory* observational study. We will NOT claim these stages as "Laws" unless we achieve clean statistical separation between stages using a held-out probe accuracy criterion.
