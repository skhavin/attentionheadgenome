# HeadGenome Phase 2: Engineering Proof & Causal Necessity

This document outlines the planned execution roadmap to solidify the HeadGenome taxonomy, resolve existing empirical gaps, and elevate the work to a publication-ready standard with strong hardware and scientific backing.

## Week 1: The Engineering Hook (Robustness & Efficiency)

*Goal: Prove the taxonomy is robust, not hand-tuned, and the HeadGenome compiler generalizes as a "No-Training Compiler" across architectures.*

### 1. Robust Entropy-Collapse (Addressing Gap 1 & 3)
- **Expand Prompt Dataset:** Increase the synthetic prompt pairs from 20 to 50 (or 100). This provides statistical defensibility against cherry-picking concerns.
- **Threshold Sensitivity Analysis:** Plot the number of classified retrieval/induction heads across a sliding threshold window (e.g., $\Delta \in [0.20, 0.45]$).
- **Deliverable:** A stability curve demonstrating that the emergence of retrieval heads is not an artifact of a magic threshold (e.g., 0.30) but a persistent structural property.

### 2. The Diffuse Retrieval Test (Addressing Gap 4)
- **Llama-3.2-1B Deep Dive:** Re-evaluate Llama-3.2-1B at lower thresholds ($\Delta = 0.20$ and $0.15$). 
- **Hypothesis:** If 10-15 heads suddenly appear at $\Delta=0.20$, Llama distributes retrieval broadly (diffuse retrieval). If counts remain near zero, Llama genuinely lacks pure retrieval mechanics in favor of pure induction.

### 3. Cross-Architecture KV Eviction (Addressing Gap 2)
- **Universal Compiler Test:** Run the `phase4/step1_routing_policy.py` benchmark on GPT-2 Medium and Qwen-2.5-0.5B.
- **Deliverable:** Prove that the static, weight-based eviction policy (which achieved a 13x perplexity improvement on Llama-1B) preserves context and defeats StreamingLLM uniformly across MHA and GQA architectures.

### 4. Decode-Time Routing & Scaling Curves (The NVIDIA Hook)
- **Theoretical Time Complexity Curve:** Generate a theoretical curve showing prefill and decode O(N) scaling.
- **Decode-Time Routing Implementation:** Apply HeadGenome identity-based masking during text generation:
  - **Sink Heads:** $O(1)$ computation (attend to BOS/early tokens only).
  - **Local Heads:** $O(W)$ computation (sliding window attention).
  - **Retrieval/Induction Heads:** $O(N^2)$ computation (full sequence attention preserved).
- **Deliverable:** Projected and empirical FLOP savings (estimated 25-40%) on local and sink heads during decoding, serving as the blueprint for a TensorRT-LLM integration.

---

## Week 2: The Scientific Proof (The Causal Necessity Test)

*Goal: Move from observational correlation to causal proof through zero-training, fast-execution hardware ablations.*

To prove the taxonomy is a mechanistic law rather than curve-fitting, we will execute targeted forward-pass ablations (zeroing out the output projections $W_O$ of specific head species) on a small, fast-eval dataset (under 10 mins per test).

### 1. Retrieval Ablation $\rightarrow$ Needle-in-a-Haystack (NIAH)
- **Method:** Zero out the ~3% of heads classified as "Retrieval".
- **Expected Outcome:** Watch NIAH accuracy drop catastrophically to near 0%, proving these specific heads are exclusively responsible for long-range factual recall.

### 2. Sink Ablation $\rightarrow$ Attention Sink Stability
- **Method:** Zero out heads classified as "Sink" (or remove the BOS token KV cache for these heads).
- **Expected Outcome:** Watch streaming perplexity explode to NaN after 1000+ tokens, proving these heads absorb excessive attention mass to prevent softmax collapse.

### 3. Induction Ablation $\rightarrow$ Prefix-Completion
- **Method:** Zero out heads classified as "Induction".
- **Expected Outcome:** Measure a massive drop in the induction score on a targeted prefix-completion task (e.g., repeating a random sequence of letters), proving they drive pattern-locking.

### 4. Local Ablation $\rightarrow$ Standard Perplexity
- **Method:** Zero out the massive block of "Local" heads.
- **Expected Outcome:** Measure standard perplexity on WikiText. PPL should degrade (as local syntax is broken) but not explode to NaN, proving these heads manage continuous linguistic fluency rather than discrete reasoning or stability.

---
*Status: Awaiting review and green signal to begin Phase 2 execution.*
