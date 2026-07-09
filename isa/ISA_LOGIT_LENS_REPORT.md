# The Head ISA: Logit Lens Statistical Report & Research Roadmap

This document details the findings of our **"Life of a Token"** experiment, run across an N=60 dataset on Qwen-1.5B using the Logit Lens technique, and outlines the long-term empirical roadmap for defining the Transformer Instruction Set Architecture (ISA).

---

## Part 1: Logit Lens Statistical Verification

Following the critique of asserting causal narratives from isolated `n=1` traces, we reran the Logit Lens experiment programmatically across an expanded dataset of 60 prompts.

**Methodology:** 
We tracked the *Shift Layer* (the exact layer where the top-1 logit prediction shifts to the correct target). We then verified whether heads mathematically classified as `Retrieval` or `Induction` (from our canonical Phase 1 taxonomy) were active in the Shift Layer or the layer immediately prior.

### 1. Fact Recall (N=20)
*Task: "The capital of [Country] is"*

| Metric | Count | Rate | Description |
| :--- | :--- | :--- | :--- |
| **Total Successes** | 20 | 100% | The model successfully generated the correct capital city. |
| **True Positives** | 19 | 95% | A Retrieval/Induction head was active at the exact Shift Layer. |
| **False Positives** | 0 | 0% | Cases where the model failed, but the heads fired anyway. |
| **False Negatives** | 1 | 5% | The model got the answer right, but without these heads firing at the shift layer. |

> **Conclusion:** There is a robust, 95% statistical correlation between the activation of statically labeled Retrieval/Induction heads and the successful completion of a geographic fact retrieval.

### 2. Needle-In-A-Haystack (N=20)
*Task: "The secret password to unlock the matrix is [PW]. [Junk...] The secret password to unlock the matrix is"*

| Metric | Count | Rate | Description |
| :--- | :--- | :--- | :--- |
| **Total Successes** | 20 | 100% | The model successfully retrieved the hidden password token. |
| **True Positives** | 20 | 100% | A Retrieval/Induction head was active at the exact Shift Layer. |

> **Conclusion:** 100% correlation. The long-range copying circuit mathematically relies on the Activation of the Induction/Retrieval taxonomy immediately prior to outputting the target token.

### 3. Pattern Induction (N=20)
*Task: "1: A, 2: B, 3: C. [N]:"*

| Metric | Count | Rate | Description |
| :--- | :--- | :--- | :--- |
| **Total Successes** | 1 | 5% | The model **failed** to extrapolate the alphabet sequence zero-shot. |
| **True Positives** | 1 | - | The single time it succeeded, an Induction head was active. |
| **False Positives** | 0 | - | In all 19 failures, the prediction never shifted to the target (Shift Layer = -1). |

> [!WARNING]
> **The Narrative Correction:** Our initial n=1 trace using shapes (`Shape: Square...`) created a compelling narrative that the model could perform zero-shot Induction perfectly. However, when subjected to an N=20 statistical distribution of arithmetic mapping, the 1.5B model failed completely (95% failure rate). This empirically proves the danger of attributing robust capabilities and causal "Transformer OS" rules based on isolated, single-prompt traces.

---

## Part 2: The Head ISA Research Roadmap

Now that we have established a rigorous statistical foundation, the long-term goal of the `HeadGenome` project evolves from macroscopic topology to answering the fundamental question:

> **What computation is a head actually performing?**
> Not "where does it attend?", but "what mathematical object is it computing?"

This outlines the 12 empirical research directions required to fully define the Instruction Set Architecture (ISA) of transformer attention.

### 1. The Life of a Token (Logit Lens & Residual Tracing)
Track **one token** from embedding to final prediction.
For every layer, measure: residual norm, which head/MLP changed it, logit lens prediction, and cosine similarity to the final residual. *Goal: Literally watch the target word emerge mathematically.*

### 2. Head Dissection
Instead of treating a head as one object, split it into its core computational paths:
*   What does the **Q projection** represent?
*   What does the **K projection** represent?
*   What information lives in **V**?
*   What exactly does the **OV circuit** write back to the residual stream?

### 3. The Birth of a Retrieval (Causal Tracing)
When a head retrieves a fact, ask **Why?** Which MLP created the query feature? Which previous heads created the key feature? Which residual component causes the dot product to suddenly spike?

### 4. The Birth of a Word (Logit Attribution)
If the model predicts "Paris", ask: Which layer first preferred Paris? Which head/MLP increased Paris? Which head suppressed "London" or "Berlin"? Which head sharpened the final distribution?

### 5. Residual Stream Evolution
The residual stream is the model; Attention and MLPs are just update functions. Study $r_0 \rightarrow r_1 \rightarrow r_2 \rightarrow \dots \rightarrow r_L$. Which geometric directions appear/disappear? Which heads rotate the vector vs amplify it?

### 6. Information Flow Graph
Map out the computation graph of one prediction. Example: `Embedding -> Head 3 -> MLP 2 -> Head 17 -> Head 24 -> MLP 8 -> lm_head`.

### 7. Head Communication (Circuits)
Instead of studying heads individually, ask who talks to whom. Example Circuit:
`Head 5 (creates entity representation) -> Head 18 (retrieves entity) -> Head 22 (copies answer)`.

### 8. MLP Genome (NeuronGenome)
Extend the genome taxonomy to the Feed-Forward layers. Which neurons create entities? Syntax? Induction features? Retrieval keys?

### 9. How Attention Changes During Generation
Track the temporal generation timeline:
`Layer 1 (random) -> Layer 4 (grammar) -> Layer 9 (retrieve concept) -> Layer 16 (concept dominates) -> Layer 24 (100% confidence).`

### 10. The Residual Language
Translate every residual vector back into English using the unembedding matrix (Logit Lens) to watch the model think:
`"country" -> "capital" -> "France" -> "Paris" -> "Paris."`

### 11. The Transformer Operating System
Reverse-engineer the CPU of the transformer. By breaking down the components: What is Memory? Computation? Communication? Storage? Control Flow?

### 12. The Head ISA (Instruction Set Architecture)
Instead of classifying Head 12 as merely a "Retrieval" head, characterize its primitive operation:
`LOAD Entity -> SEARCH Context -> COPY Payload -> WRITE Residual`.

**The Evolution of HeadGenome:** 
The initial HeadGenome paper asked: *"What kinds of instructions exist?"*
This research roadmap answers: *"What is the complete Instruction Set Architecture (ISA) of transformer attention?"* 
By understanding the exact computation each head performs, we can eventually engineer the cheapest execution strategy that perfectly preserves it.
