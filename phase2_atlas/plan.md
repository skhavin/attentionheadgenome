# Phase 2: The HeadGenome Atlas (Level-4 Anatomy)

## Core Philosophy
Attention head identity is not a simple scalar property (e.g., entropy collapse). True mechanistic identity factorizes into four distinct levels of behavior:
1. **Where it looks (Selector Geometry / QK):** What structural or semantic signals attract the head?
2. **What information it reads (Value Content / V):** What payload does it extract from the target?
3. **What it writes (Write Effect / OV):** What direction does it inject into the residual stream?
4. **Whether it matters (Causal Contribution):** Do later heads, MLPs, or the unembedding matrix actually use this write?

**Goal:** We are replacing unsupervised attention-pattern clustering with a comprehensive **behavioral atlas**. This transforms our classification from *"These attention patterns look geometrically similar"* to *"This head performs this exact function because its QK selector, OV writer, output norm, and downstream causal effect behave specifically like this."*

---

## 1. The Head Profile (JSON Output Format)
Every experiment must output heavily structured JSON files. We will generate a complete functional dictionary of transformer attention heads. For every important head, we will create a card matching this schema:

```json
{
  "model": "Qwen-2.5-0.5B",
  "layer": 12,
  "head": 7,
  "class_label": "Retrieval",
  "vq_ratio": 1.31,
  "locality_ratio": 0.18,
  "entropy_profile": {
    "mean_entropy": 0.54,
    "delta_collapse": 0.82
  },
  "qk_behavior": {
    "top_attended_pos": ["PROPN", "NUM"],
    "top_dependency_target": ["appos", "nmod"],
    "top_semantic_category": ["names", "dates", "capitals"]
  },
  "ov_behavior": {
    "output_norm": "high",
    "direct_logit_effect": "medium"
  },
  "causal_effect": {
    "ablation_niah_drop": 0.45,
    "failure_cases": ["attends_wrong_matching_entity"]
  },
  "nearest_sibling_heads": ["12_5", "14_2"]
}
```

---

## 2. Experimental Pillars

### Pillar 1: Attention Target Geometry (Where it looks)
*   **Distance Profiling ($t - j$):** Plot each head's relative distance profile (Sink curve, Local diagonal curve, Retrieval spikes, Induction offset patterns).
*   **RoPE Sensitivity:** Test whether heads depend on relative phase by shifting prompts, padding, or evicting tokens, mapping which functional classes are most fragile to RoPE compression.
*   **GQA Group Analysis:** For Llama/Qwen, analyze query groups sharing K/V. Does a single K/V group support multiple distinct query roles? Is specialization smeared across sibling Q-heads?

### Pillar 2: QK vs OV Separation (The Engine)
*   **QK Circuit (The Search Engine):** Use activation patching, logit lens on query/key directions, and nearest-neighbor tokens in Q/K space to determine exactly what makes a head assign high attention (e.g., universal BOS attraction vs semantic matching).
*   **OV Circuit (The Writer):** Use Direct Logit Attribution (DLA) ($O_h \cdot W_U[\text{target}]$) to measure how much a head pushes the correct answer token. Separate "heads that look useful" from "heads that are causally useful."
*   **Attention vs Contribution Mismatch:** Plot attention mass vs logit contribution. (High attention / Low contribution = Locator; Low attention / High contribution = Integrator; High attention / High contribution = Direct Retrieval).
*   **Value-Vector Content:** Cluster V outputs to see if Retrieval heads locate facts while Induction heads carry the actual answer-bearing values.

### Pillar 3: Semantic and Grammatical Specialization
*   **Universal Dependencies (Grammar Mapping):** Use English treebanks to map exactly what grammatical roles heads attend to (subjects, verbs, objects, determiners, cross-dependency arcs).
*   **Prompt-Regime Switching:** Expand to measure conditional head identity—e.g., a head that is Local in normal text, Retrieval in code, Induction in repetition, and Sink under uncertainty.
*   **Token-Type Atlas:** Map head triggers across names, dates, URLs, rare tokens, punctuation, brackets, and instruction markers.

### Pillar 4: Circuit Architecture & Causal Handoffs
*   **Head-to-Head Dependency Graphs:** Use iterative patching to build a real circuit diagram. If we ablate a Retrieval head, does the downstream Induction head lose its alignment?
*   **Residual Stream "Handoff" Analysis:** Track where information moves via linear probes/logit lens (Embedding $\to$ Early Heads $\to$ Retrieval $\to$ Residual $\to$ Induction $\to$ MLP $\to$ Logits).
*   **Early vs Late Induction:** Mechanistically prove that Early Induction finds the previous matching prefix (A), while Late Induction copies the payload (B).
*   **Sink Head Falsification:** Replace BOS, move BOS, and remove BOS to measure entropy explosions and prove whether Sinks exist for null-space routing or softmax stabilization.
*   **Perplexity Illusion Decomposition:** Determine exactly which heads preserve local perplexity vs which heads preserve long-range retrieval routing.

---

## 3. The Dataset Stack
We will use three heavily targeted dataset types:
1.  **Natural English Corpus:** (WikiText-103, PG-19) for local grammar, long text, and calibration.
2.  **Universal Dependencies Treebanks:** For strict tracking of grammatical components (nsubj, obj, amod, root) to scientifically measure syntax specialization.
3.  **Controlled Synthetic Grammar Suite:** Bespoke templates where we know *exactly* which token the head *should* attend to. 
    *   *Examples:* `Alice gave Bob the book. Bob gave Alice the book.` / `The password is BLUE7. Later, what is the password?`

---

## 4. Expanded Deeper Behavioral Laws (New Frontiers)
To expand beyond the initial master list, we will investigate these deeper phenomena and establish formal laws governing transformer behavior:

1.  **The Structural V/Q Scaling Law:** Structural observations suggest that deeper heads become increasingly Value-dominant (high V/Q weight norm ratios). We will test this behaviorally: Do high V/Q heads actually produce larger output norms ($||O_h||$) and causally exert larger Direct Logit Attribution effects? Proving this bridges structural weight norms with causal behavioral power.
2.  **The Retrieval-Induction Co-Gating Law:** The Phase 5 co-gating results proved that Retrieval (locator) and Induction (writer) heads must act together to solve NIAH. We will map the exact Boolean geometry of this circuit: If a Retrieval head's $QK$ locator strength drops by $x\%$, does the downstream Induction head's $OV$ write strength degrade linearly, or does it trigger a catastrophic binary collapse (a strict logical AND gate)?
3.  **The Depth-Stratified Specialization Law:** The Master Report shows Sink/Local heads dominating early layers, while Retrieval/Induction concentrate in mid-to-late layers. We will measure whether this macro-stratification is an immutable law of gradient descent by tracking head emergence across training checkpoints. Does the "grammar layer" strictly have to converge before the "retrieval layer" can form?
4.  **Polysemantic Multiplexing:** Do single attention heads multiplex multiple behaviors (e.g., Local and Retrieval) depending on orthogonal subspaces of the Query vector? We will apply Sparse Autoencoders (SAEs) directly to the output of high-variance regime-switching heads.
5.  **Cross-Layer Feedback Routing:** Do deep integration heads write specific directions into the residual stream that are strictly intended to be read by the MLP layers, or do they write directions that modify the KV-caching behavior for subsequent multi-turn generation?
6.  **Attention Mass Quantization:** Is attention mass continuously distributed, or does it collapse into discrete, quantized states under extreme scale and deep layers? 
7.  **The Anti-Copy Inhibition Circuits:** Mechanistically reverse-engineer the "hyper-diagonal" outlier heads to prove whether they function as negative suppression gates (preventing exact string copying when abstract reasoning is required).
8.  **The Emergence Threshold Law:** Why do certain behaviors (like Induction) fail catastrophically below a critical scale (e.g., <1B parameters), while others (like sinks) are robust? We will test the hypothesis that Induction requires a minimum number of heads per layer to form a stable recurrent loop, while Sinks are supported by the raw matrix dimensions. We will perform targeted ablation of "late" Induction heads in a 0.5B model to search for the sharp phase transition where the Induction circuit collapses.
9.  **The Attention-MLP Symbiosis Law:** Attention routes information, but MLPs process it. We will map the exact handoff: Do specific Integration Heads exist purely to route information into specific MLP feature directions? (Testing if ablating a single head destroys the activation of a corresponding MLP concept neuron).
10. **The Residual Stream Erasure Law (The Forgetting Mechanism):** Attention heads don't just add; they can subtract (via negative vector projections). Is there a distinct class of "Erasure Heads" that proactively zero-out stale information from the residual stream (e.g., erasing the previous subject vector once a new sentence begins) to prevent context pollution?
11. **The Softmax Saturation Law:** Do different classes of heads operate in different mathematical regimes? We will test if Retrieval heads rely on extreme softmax saturation to function (acting as hard binary gates), while Local heads operate evenly in the pre-softmax, distributed regime.
12. **The Representation Superposition Breakdown Law:** How does a head behave when a token is polysemantic (e.g., "Apple" the fruit vs the company)? Do Retrieval heads resolve the superposition in the QK phase before extracting the value, or do they blindly copy the superposed V-vector and force the downstream MLP to resolve it?
13. **The Long-Context Degradation Law (Attention Dilution):** As the context window expands past 128k tokens, does the QK dot product for Retrieval heads linearly decay due to softmax dilution across thousands of tokens, or do true Retrieval heads possess a mathematical mechanism to maintain a constant signal-to-noise ratio regardless of $N$?
14. **The Multi-Hop Reasoning Law (Transitive Routing):** When reasoning $A \to B \to C$, is there a specific circuit class (Transitive Heads) that reads the Value of token B but uses the Query of token C, thereby mathematically bridging two independent facts in a single forward pass?
15. **The Positional Interpolation Law (RoPE Extrapolation):** How do specific head classes behave when pushed past their trained context limit? Does a Local head "smear" its syntactic window, or does an Induction head completely lose its phase-tracking alignment?
16. **The KV Cache Mini-Sink Law:** Beyond the BOS token, why do some arbitrary tokens (like commas or newlines) accumulate massive attention mass over long distances? Do these act as structural "mini-sinks" for local chunking, mathematically resetting the attention state for the next structural clause?
