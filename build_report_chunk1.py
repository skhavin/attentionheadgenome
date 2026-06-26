import os

md_content = """# The HeadGenome Master Report: A Structural and Behavioral Taxonomy of Attention Heads

**Date:** June 2026
**Models Analyzed:** GPT-2 Medium (355M), Qwen-2.5-0.5B, Qwen-2.5-1.5B, Llama-3.2-1B
**Total Heads Analyzed:** 1,568 Attention Heads

---

## Executive Summary

The HeadGenome Project bridges the gap between transformer interpretability and systems engineering. By applying rigorous empirical testing across multiple model architectures, we have mapped the functional ecology of attention heads. We discard the traditional view of attention heads as isolated, homogenous mathematical operations. Instead, we propose that attention heads occupy a **low-dimensional developmental manifold**. Static geometry predicts a head's developmental stage, dynamic probes identify its functional specialization, and sparse compression algorithms can heavily exploit this specialization for $O(N^2)$ prefill and decode optimization.

This document serves as the comprehensive, ground-truth record of all findings, mathematical formulations, execution scripts, and output data utilized in this research.

---

# PART I: Theoretical Foundation & Transformer Mechanics

Before defining the taxonomy, it is critical to formalize the mechanical structures, datatypes, and exact mathematical operations that govern the models studied.

## 1.1 The Anatomy of an Attention Head

An attention head in a standard Transformer model maps an input sequence of hidden states $X \in \mathbb{R}^{N \\times d_{model}}$ to an output sequence $O \in \mathbb{R}^{N \\times d_{head}}$, where $N$ is the sequence length.

**Exact Operations:**
1. **Linear Projections:** The input $X$ is multiplied by three learned weight matrices:
   * Query Projection: $W_Q \in \mathbb{R}^{d_{model} \\times d_{head}}$
   * Key Projection: $W_K \in \mathbb{R}^{d_{model} \\times d_{head}}$
   * Value Projection: $W_V \in \mathbb{R}^{d_{model} \\times d_{head}}$
   
   Yielding $Q = X W_Q$, $K = X W_K$, and $V = X W_V$.

2. **Pre-Softmax Attention Scores:**
   $S = \\frac{Q K^T}{\\sqrt{d_{head}}}$
   Where $S \in \mathbb{R}^{N \\times N}$ is the raw, unnormalized attention score matrix. A causal mask is applied such that $S_{i, j} = -\\infty$ for $j > i$.

3. **Post-Softmax Attention Weights:**
   $A = \\text{Softmax}(S, \\text{dim}=-1)$
   $A \in \mathbb{R}^{N \\times N}$ represents the probability distribution of attention mass. $\\sum_{j \le i} A_{i, j} = 1$.

4. **Value Aggregation and Output:**
   $O_{head} = A V$
   Finally, all head outputs are concatenated and multiplied by an output projection matrix $W_O \in \mathbb{R}^{d_{model} \\times d_{model}}$.

**Datatypes in Execution:**
All structural analysis in this project was conducted using FP32 (Float32) extracted parameters or FP16 (Float16) depending on the huggingface checkpoint. Dynamic forward passes were executed utilizing `torch.float16` or `torch.bfloat16` to fit within standard VRAM constraints, particularly for Llama-3.2-1B and Qwen-2.5-1.5B models.

## 1.2 Multi-Head Attention (MHA) vs. Grouped Query Attention (GQA)

The functional ecology of heads is heavily influenced by the routing architecture.

* **MHA (GPT-2 Medium):** 24 layers, 16 heads. Each head has its own isolated $W_Q$, $W_K$, and $W_V$. This allows for extreme, isolated specialization (e.g., highly specific single-head retrieval).
* **GQA (Qwen-2.5, Llama-3.2):** GQA restricts the number of Key/Value heads. For example, Llama-3.2-1B has 32 Query heads but only 8 KV heads. This means 4 Query heads must share the same $K$ and $V$ representations.
* **Impact on Specialization:** As proven in `outputs/phase6/llama_diffuse_threshold.json`, GQA forces "diffuse" specialization. A single query head cannot easily hijack the KV pathway to act as a pure retrieval head without impacting its 3 sibling heads.

## 1.3 Position Embeddings: Absolute vs. RoPE

* **Absolute Embeddings (GPT-2):** A learned embedding vector is added to the token embedding at each absolute index $i$. Evicting tokens from the KV cache shifts the absolute indices of all subsequent tokens, causing catastrophic perplexity degradation (measured in `outputs/phase4/routing_policy_results.json`).
* **Rotary Position Embeddings / RoPE (Llama, Qwen):** Position is encoded by rotating the $Q$ and $K$ vectors based on their relative distance $(i - j)$. This permits KV Cache eviction because the relative distances between remaining tokens are preserved.

---

# PART II: Static Geometry vs. Dynamic Behavior

A core hypothesis of the HeadGenome project was that attention heads could be classified by analyzing their frozen weight matrices. This proved to be mathematically false, leading to the first major empirical finding.

## 2.1 Finding 1: Histogram Invisibility
**The Observation:** Static Weights $\\neq$ Real-Time Workflow. Mapping the functional ecology of a Transformer requires a second axis of dynamic, synthetic entropy-collapse probing.

### Methodology & Execution
* **Script:** `paper_analysis_suite.py` and `phase2/step2_clustering.py`
* **Output Data:** `outputs/phase8_paper_suite/statistical_suite_results.json` and `outputs/phase2/cluster_metrics.json`

We extracted static weight matrices for all heads in GPT-2 and computed the Singular Value Decomposition (SVD) of the $W_Q, W_K, W_V, W_O$ matrices, alongside Frobenius weight norms. We then performed unsupervised K-Means clustering ($K=4$).

### Results
When these geometric clusters were cross-referenced against ground-truth behavioral labels (obtained via dynamic probing), the clusters were completely flattened:
* **Cluster C0 (n=188):** 10 Sink, 155 Local, 3 Retrieval, 20 Induction.
* **Cluster C2 (n=81):** 3 Sink, 52 Local, 6 Retrieval, 20 Induction.

**Conclusion:** Retrieval and induction heads are "histogram-invisible" to standard weight clustering. They possess static footprints identical to standard local heads. To classify a head, we must measure its dynamic response to structured prompts.

## 2.2 Finding 2: The $||V|| / ||Q||$ Developmental Scaling Law
**The Observation:** Transformers utilize a temporal structural pipeline. Early layers act as query-dominant "locators", while deep layers mature into value-dominant "payload delivery systems."

### Mathematical Formulation
For every head, we calculate the Frobenius norm of its combined Query and Value projection matrices relative to the model dimension:
Ratio = $||W_V||_F / ||W_Q||_F$

### Methodology & Execution
* **Script:** `paper_analysis_suite.py` and `plot_developmental_curve.py`
* **Output Data:** `outputs/phase8_paper_suite/statistical_suite_results.json`

### Empirical Results
We correlated this ratio against the head's relative depth in the network ($layer\_idx / total\_layers$):
* **GPT-2 Medium:** $r = 0.681$
* **Qwen-2.5-0.5B:** $r = 0.734$
* **Qwen-2.5-1.5B:** $r = 0.647$
* **Llama-3.2-1B:** $r = 0.635$

**Global Statistical Significance:** $p = 1.92 \\times 10^{-127}$.
This massive, cross-architectural scaling law confirms that attention heads mature systematically across depth. 

---

# PART III: The Developmental Manifold & Functional Taxonomy

Based on the V/Q scaling law and dynamic entropy measurements, we classify the functional taxonomy of attention heads. The four head types are not independent discrete circuits; they represent stable regions of a continuous developmental manifold.

## 3.1 The Metric of Dynamic Specialization: Entropy Collapse ($\\Delta$)

To measure dynamic specialization, we define Attention Entropy for head $h$ at token step $t$:
$H(A_{h, t}) = -\\sum_{j=1}^{t} A_{h, t, j} \\log_2(A_{h, t, j})$

When faced with a specific task (e.g., retrieving a hidden needle or completing a repeating pattern), a specialized head will collapse its attention mass onto a single target token, causing a massive drop in entropy relative to its baseline processing state.

We measure this as $\\Delta = H_{task} - H_{baseline}$.
* Positive $\\Delta$ (e.g., $+0.30$): The head drastically sharpens its focus (Retrieval).
* Negative $\\Delta$ (e.g., $-0.50$): The head drastically broadens its focus or changes its pattern (Induction).

## 3.2 Sink Heads (Phase 1: Infancy)
**Function:** Stable attention sinks that absorb excess attention mass when no highly relevant contextual information is present. This prevents attention dilution across random tokens, allowing the network to "ignore" irrelevant steps.
* **Execution Script:** `phase1/step2_threshold_sensitivity.py` and `phase1/step3_profile_llama.py`
* **Output Data:** `outputs/phase1/threshold_sensitivity.json`
* **Methodology:** Classified via static mass accumulation. A head is a Sink if it overwhelmingly allocates attention to the first token (BOS or start) across diverse distributions, regardless of the prompt.
* **Verification:** Causal ablation of the 15 Sink heads in GPT-2 degraded perplexity by +199.37 points (`outputs/phase5/fixed_ablation.json`).

## 3.3 Local Heads (Phase 2: The Precursor State)
**Function:** The default sliding-window routing backbone of the model. They process syntactic grammar and immediate neighboring token relationships.
* **Execution Script:** `phase1/step2_threshold_sensitivity.py`
* **Output Data:** `outputs/phase1/gpt2_mechanistic_labels.json`
* **Methodology:** Classified by neutral dynamic entropy ($\\Delta \\approx 0$). They process a rolling local window ($W \\approx 32$ to $512$). 
* **The Manifold Concept:** ~85% of all heads remain in this stable, undifferentiated state. They occupy the **branching region** of the developmental manifold.

"""

os.makedirs("outputs/final_artifacts", exist_ok=True)
with open("outputs/final_artifacts/HeadGenome_Master_Report.md", "w", encoding="utf-8") as f:
    f.write(md_content)
print("Chunk 1 written successfully.")
