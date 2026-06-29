# Phase 9: Semantic Profiling & Linguistic Universality

## Objective
The current HeadGenome taxonomy (Sink, Local, Retrieval, Induction) is defined strictly by structural mathematics (Entropy Collapse $\Delta$ and $V/Q$ norm ratios). To make the taxonomy unassailable for peer review, we must bridge the gap between **structure** and **semantics**. 

This phase will map exactly *which English words and syntactic structures* these mathematically defined heads attend to. By showing that structurally identical heads across different architectures (GPT-2, Qwen, Llama) focus on the exact same linguistic features, we will visually and statistically prove the **Architectural Universality** of the HeadGenome.

---

## 1. How Semantic Profiling Solidifies the Taxonomy

### A. Local Heads (The Grammar Engine)
* **Current Definition:** Low entropy collapse, stable sliding window. Viewed as an undifferentiated precursor pool.
* **Semantic Goal:** Prove that the "Local" pool is actually a highly organized syntactic parser. By mapping their target tokens, we expect to find specific Local heads dedicated to tracking Nouns, Verbs, or Adjectives. This proves they are processing grammar, not just blindly looking at recent tokens.

### B. Induction Heads (Visualizing the "Copy")
* **Current Definition:** Extreme negative entropy ($\Delta \leq -0.5$) during repetitive sequences.
* **Semantic Goal:** Generate a text-overlaid heatmap during a repetition task (`[A][B] ... [A] -> [B]`). The visualization will physically show the attention mass reaching back to highlight the exact `[B]` token in bright red, providing undeniable visual proof of the payload-copy mechanism.

### C. Retrieval Heads (The Semantic Router)
* **Current Definition:** Extreme positive entropy ($\Delta \geq 0.3$) during Needle-In-A-Haystack tasks.
* **Semantic Goal:** Feed a factual lookup prompt. The visualization will show 90%+ attention mass concentrated exclusively on the specific factual token (the "needle"), ignoring all surrounding grammatical fluff.

### D. Sink Heads (The Attention Dumps)
* **Current Definition:** Baseline match entropy $H_{match} < 0.1$.
* **Semantic Goal:** While GPT-2 dumps on `[BOS]`, models with RoPE (like Llama) lack a `[BOS]` token. Semantic profiling will reveal where Llama's dormant heads dump attention (e.g., periods, commas, or newline characters), proving the mechanical necessity of an "attention dump" regardless of architecture.

---

## 2. Implementation Methodology

### Step 1: The "Universality Matrix" (Text-Overlaid Visualizer)
We will abandon abstract 2D grid matrices in favor of **Text-Anchored Heatmaps**.
* **Script:** `visualize_semantic_taxonomy.py`
* **Process:** 
  1. Load `outputs/canonical_labels.json` to identify the "purest" representative heads (highest $\Delta$ magnitudes) for each class in GPT-2, Qwen-0.5B, and Llama-3.2-1B.
  2. Run specific, targeted sentences through the models (e.g., a factual lookup, a repeating pattern, a complex grammatical sentence).
  3. Extract the raw attention matrices for the last token generation.
  4. Generate an HTML visualization where the input text is highlighted based on the attention weight from that specific head.
* **Outputs:** 
  * `outputs/phase9_semantics/semantic_attention_data.json` (Raw attention maps and tokens)
  * `outputs/phase9_semantics/universality_matrix.html` (Interactive visual proof)

### Step 2: Large-Scale Dictionary/POS Audit (Future Scope)
Once the visualizer proves the concept, we can scale this up by running 1,000 sentences from WikiText-103, parsing the Part-Of-Speech (POS) tags of the maximally attended tokens, and generating a statistical bar chart of what Grammar each head specializes in.
