# Beyond Operation Geometry: Dynamic Instruction Trajectories in Transformer Residual Space

## 1. The Core Philosophy: A Shift in the Unit of Analysis
The history of mechanistic interpretability has seen a progressive shift in the fundamental "unit of analysis":
`Neuron` $\rightarrow$ `Attention Head` $\rightarrow$ `Residual Direction` $\rightarrow$ **`Residual Trajectory`** $\rightarrow$ **`Dynamic Computation`**

Our previous work (*The Residual ISA*) successfully broke away from hardware-bound instructions, proving that operations have conserved geometry in the residual stream. However, it concluded with an intentional negative result: that geometry is **necessary, but not sufficient**. Single-layer additive substitution failed. 

This paper answers **why**. 
If computation is not a static point injected at a single layer, it must be an evolving geometric object distributed across time (depth). 

## 2. The Overarching Scientific Question
> **Is the fundamental computational object in a transformer a static representation, or a dynamic trajectory through residual space?**

### Competing Hypotheses
*   **H1 (The Static Point):** Computation is encoded as a single-layer state. *(Challenged by our previous findings).*
*   **H2 (The Temporal Curve):** Computation is encoded as a multi-layer trajectory.
*   **H3 (The Distributed System):** Computation depends on a distributed interaction (trajectory plus dynamic control), meaning even pure trajectory replacement will be insufficient.

---

## 3. The Dynamic ISA Hypothesis
**Transformers do not execute fixed hardware instructions. Instead, they execute dynamic computational trajectories in residual space that emerge over multiple layers through coordinated interactions between attention and MLP modules.**

Every experiment in this research program is designed to test a single branch of this hypothesis.

---

## 4. The Experimental Roadmap

### Section 1: The Birth of Computation (Probing)
Instead of asking *where* comparison is, we ask *when* it appears. 
*   **Execution:** Train a simple linear probe at every layer (Layer 1 $\rightarrow$ N). 
*   **Goal:** Map the gradual emergence of the computational operation. Does an operation like `Comparison` spike instantly, or is it constructed gradually across 10 layers?

### Section 2: Trajectory Mapping (Manifold Extraction)
Instead of plotting one residual point (e.g., Layer 19), every prompt becomes a continuous curve.
*   **Execution:** Extract the residual stream at *every layer* for 100 prompts across 6 distinct categories (Comparison, Copy, Counting, Fact Recall, Sorting, Arithmetic).
*   **Goal:** Visualize and mathematically analyze the trajectory manifold. Do all `Comparison` prompts follow the same path? Where do they diverge from `Copy`? 

### Section 3: Trajectory Conservation (Cross-Architecture Alignment)
Do entirely different models follow the same developmental path?
*   **Execution:** Expand the Representational Similarity Analysis (RSA) from static endpoints to entire trajectories (using metrics like Dynamic Time Warping, Frechet distance, or Trajectory Kernels).
*   **Goal:** Prove that the *motion* of computation is universally conserved across Qwen, Llama, and Phi, establishing comparative neuroscience for transformers.

### Section 4: Multi-Layer Intervention (The Definitive Causal Test)
If the ISA is temporal, single-layer patching is doomed to fail. We must override the trajectory.
*   **Execution:** Instead of replacing Layer 19, replace a sliding window of layers (e.g., 16-20, or 10-25) with the target trajectory.
*   **Goal:** Test H2 directly. If injecting a continuous 5-layer trajectory successfully steers the model (where a 1-layer patch failed), we have discovered that the ISA is fundamentally temporal.

### Section 5: Generator Analysis (The Construction Process)
If the trajectory exists, *who writes it?*
*   **Execution:** Measure the exact projection contribution of every Attention Head and MLP onto the evolving trajectory direction at each layer.
*   **Goal:** We no longer care who *stores* the instruction; we care who *constructs* it. This builds a dynamic graph of the trajectory's formation.

### Section 6: Control Signals (The Activation Trigger)
If multi-layer trajectory intervention still fails (H3), what activates the trajectory?
*   **Execution:** Search for precursor features that consistently precede the emergence of the computational trajectory (e.g., a "Question" control feature that triggers the "Comparison" geometry).
*   **Goal:** Explain why a trajectory is necessary but potentially gated by an independent control signal.

---

## 5. Execution Strategy: The Highest Expected Payoff
The immediate priority (Experiment 1 & 2) is **Section 2 (Trajectory Mapping) + Section 4 (Multi-Layer Intervention)**. 

1. Generate ~100 prompts for the 6 core categories.
2. Save the residual stream at *every layer* for every prompt.
3. Compute the average trajectory for each category and measure within/between category similarities.
4. Perform multi-layer trajectory replacement over sliding windows (3, 5, 7 consecutive layers).

This directly answers the cleanest scientific question: **Is computation represented as a trajectory rather than a static state?**
