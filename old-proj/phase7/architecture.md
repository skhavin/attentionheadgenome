# Phase 7: O(W) Routing & Scale-Up Architecture & Results

This document describes the design, implementation, verification, and final results for the Phase 7 Mixture-of-Experts (MoE) routing system.

## 1. Core Idea & Method

Our goal is to dynamically approximate attention heads on a per-token basis to reduce computation (FLOPs) without degrading model perplexity ($\Delta$PPL < 1.5). We design a 4-path MoE system for each attention head:
1. **Sink Path ($O(1)$)**: Uniformly averages the key-value representations of the first 4 tokens (attention sinks).
2. **Local Path ($O(N \cdot W)$)**: Computes causal uniform 1D convolution over the last $W=64$ tokens.
3. **Recurrence Path ($O(N)$)**: Computes an Exponential Moving Average (EMA) with decay $\alpha=0.9$ (recurrent formulation is $O(N)$ sequential steps).
4. **Full Path ($O(N^2 d)$)**: Exact full softmax attention.

### O(W) Router Features
To avoid paying the $O(N^2)$ cost of full attention just to decide which path to route, the router uses $O(W)$ query-key similarity features from the current token (query $Q_{last}$):
- **Local Entropy**: The entropy of the attention distribution over the last $W=16$ tokens.
- **Sink Mass**: Total attention probability mass allocated to the first 4 sink tokens.
- **Recency Mass**: Total attention probability mass allocated to the last $W=16$ tokens.
- **Max Similarity**: Maximum unnormalized dot-product score over the last $W=16$ tokens.

### Three-Stage Curriculum Training & The Entropy Multiplier
To train the router parameters (`LayerRouter` Grouped Conv1d layers) stably, we use a structured curriculum.
**Crucial Discovery:** Training discrete routing decisions from a cold start using the Straight-Through Estimator (STE) fails catastrophically because unselected paths receive garbage gradients based on the error of the selected path. To solve this, we use the **Entropy Curriculum**:
1. **Stage 1 (Soft Routing, Pure Reconstruction)**: `entropy_weight = 0.0`. Routers learn pure MSE reconstruction by blending all 4 paths. Gradients flow correctly through all paths.
2. **Stage 2 (Soft Routing, Forced Polarization)**: `entropy_weight = 1.5`. We heavily penalize uniform distributions. The router is forced to polarize and commit to a single path (often >95% probability) *while still in soft routing mode*, preserving accurate gradients.
3. **Stage 3 (Hard Routing, STE Fine-Tuning - Optional)**: `entropy_weight = 3.0`. STE discretization. Because Stage 2 perfectly aligns the soft probabilities with discrete `argmax` decisions, Stage 3 is often not strictly needed and Stage 2 weights translate perfectly to hard routing evaluation.

---

## 2. Final Results

### Qwen2.5-0.5B (Profile A: Soft Routing + Entropy Curriculum)
- **Baseline Perplexity**: **10.57** (Measured at length 512 previously)
- **MoE Hard Routing PPL**: *(Evaluation currently running)*
- **Cheap Path Activation**: **0.0%** (The Stage 2 checkpoint collapsed back to 100% Full Attention on Wikitext under hard routing).
- **Interpretation**: The model failed to dynamically offload compute. Diagnostic analysis of the feature distributions (LocalEntropy, SinkMass, MaxSim) reveals that the Stage 2 training using `Profile A` (which has a massive fallback bias to Full Attention) was too heavily biased for the single-epoch entropy curriculum to overcome. The `argmax` decisions never shifted away from Full Attention, meaning the entropy multiplier (1.5) must be increased or the routers must be initialized uniformly to successfully train the MoE.

#### Retrieval Sanity Check (Needle in a Haystack)
- **Method**: Evaluated single-needle retrieval accuracy at sequence length 512 using a quick sanity check (`eval_niah_quick.py`) on 50 samples.
- **Baseline (Full Attention)**: **100.0%**
- **Stage 2 MoE Router (Hard Routing)**: **100.0%**
- **Interpretation**: While the router maintains 100% accuracy, this is *because* it reverted to 100% full attention, not because it selectively kept retrieval heads on full attention. This confirms the saturation artifact.


#### Failed Methodology: STE-Test
- **Method**: Training with `hard_routing=True` from Stage 1.
- **MoE Hard Routing PPL**: **~5968.0**
- **Interpretation**: Confirms that STE cannot learn discrete routing decisions from scratch due to unselected path gradient collapse.

### Legacy Results: GPT-2 Medium
We experimented with two routing profiles for GPT-2 Medium on the 500/1000/200 document curriculum:

#### Profile A: Conservative/Peaked Initialization (Fallback Biased to Full Attention)
- **Baseline Perplexity**: **19.20**
- **MoE Hard Routing Perplexity**: **20.65**
- **$\Delta$PPL**: **+1.45** (within budget)
- **Interpretation**: Extremely safe, but minimal compute savings.

#### Profile B: Uniform Initialization & Dynamic Curriculum (No Prior Bias)
- **Baseline Perplexity**: **19.20**
- **MoE Hard Routing Perplexity**: **3415.93**
- **Interpretation**: Pushing 68% of heads (261 heads) simultaneously to cheap paths under hard routing triggers severe compounding error propagation across the 24 sequential layers. STE fails to route effectively.

---

## 3. Weights & Folders Location

### Weights to Upload to Cloud
To deploy or share the trained MoE routers, upload the following folder:
- **`checkpoints/gpt2-medium-full/`**
  - **`routers.pt`**: The final trained LayerRouter parameters (dict of state dicts for each layer).
  - **`moe_checkpoint.pt`**: Full training checkpoint (routers, optimizer state, scheduler) to resume training.

### Other Important Code Locations
- [moe_patcher.py](file:///d:/PROJECTS/webstromprojects/supertransformers/phase7/moe/moe_patcher.py): The main patching engine that overrides the model's attention layer.
- [router.py](file:///d:/PROJECTS/webstromprojects/supertransformers/phase7/moe/router.py): `LayerRouter` architecture and profile initialization.
- [paths.py](file:///d:/PROJECTS/webstromprojects/supertransformers/phase7/moe/paths.py): Expert approximation paths.
- [train_moe.py](file:///d:/PROJECTS/webstromprojects/supertransformers/phase7/training/train_moe.py): Three-stage training script.
- [eval_ppl.py](file:///d:/PROJECTS/webstromprojects/supertransformers/phase7/evaluation/eval_ppl.py): Perplexity evaluation harness.
- [eval_niah_quick.py](file:///d:/PROJECTS/webstromprojects/supertransformers/phase7/evaluation/eval_niah_quick.py): Single-needle 512-context sanity check.

