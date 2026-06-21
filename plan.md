# HeadGenome - The Attention Atlas 
## by S Khavin

1. The Core Paradigm Shift: From Stamp Collecting to Genomics
Mechanistic interpretability currently treats attention heads like 19th-century biology—discovering a "species" (e.g., an induction head or sink head), naming it, and moving on. HeadGenome shifts this from a descriptive science to a predictive, systems-driven architecture.

The Old Claim: "Nobody knows what attention heads do" (False).

The Real State of the Field: We know what individual heads do in specific models, but we lack a clean, cross-architecture, universal taxonomy.

The HeadGenome Pivot: Instead of asking "What does this specific head do?", you are asking: "What universal developmental laws dictate how, why, and where specific head behaviors emerge across entirely different models?"

2. The Four Pillars of the HeadGenome Hypothesis
To elevate this from a casual observation to a bulletproof systems framework, your direction relies on four interconnected hypotheses:

Hypothesis 1 (Universal Ecology): Across disparate models (Meta, Alibaba, Mistral AI), attention heads converge into a small, finite set of functional classes (Sink, Local, Retrieval, Induction, Composition).

Hypothesis 2 (Statistical Signature): These functional classes can be completely identified using raw attention statistics (e.g., sink mass, attention entropy, average attention distance) rather than expensive manual probing.

Hypothesis 3 (Predictive Approximation): A head's functional class strictly predicts which cheap runtime approximation strategy (e.g., local window, token eviction, sliding window, or full attention) preserves its perplexity.

Hypothesis 4 (Universal Generalization): A routing policy derived from these rules can generalize to an unseen model trained on an unseen dataset without any calibration or retraining.

3. The Clue: The ~300 Document Convergence Phenomenon
Your KV eviction data showed that prototype head clusters stop changing after processing roughly 300 WikiText documents across GPT-2, Qwen, and Llama. This is a massive clue, but it must be interpreted accurately:

What it does NOT prove: It does not prove that Layer 10, Head 2 is identical in every model.

What it DOES prove: It proves that attention behavior space is low-dimensional. Language presents a finite set of structural problems (tracking pronouns, retrieving far-off facts, anchoring to syntax). Because the problem space is small, different models independently evolve the exact same structural tools.

The Analogy: It is evolutionary convergence—like nature independently inventing the eye in completely different species because the physics of light remain constant.

4. The Ultimate Value Proposition: A Training-Free Compiler
The reason this is a systems breakthrough and not just an interpretability project is the endgame. If the developmental laws of attention are invariant, you can build a Universal Attention Compiler:

New Unseen Model ➔ Analyze Head Statistics ➔ Predict Head Species ➔ Apply Optimal Approximation Route ➔ Instant Faster Inference
This bypasses the massive bottlenecks of current inference optimization:

No profile-driven tuning: No need to run hardware-heavy profiling passes.

No calibration datasets: No risk of overfitting to a specific calibration slice.

Zero-shot optimization: The runtime policy works right out of the box because it maps directly to the model's evolved "DNA."

5. The Scientific Roadmap (Phases 1–4)
Because the risk is high (i.e., architectures like RoPE vs. absolute positions might fracture the "genome"), you shouldn't engineer the router first. The router is built on sand if the universal laws don't hold. The execution should follow a rigorous scientific sequence:

Phase 1: The Head Atlas (Data Gathering)
Profile a matrix of diverse models (Llama, Qwen, Mistral, Gemma) across distinct parameters: sink mass, attention entropy, average distance, and retrieval sensitivity.

The Goal: Prove that the exact same behavioral clusters emerge in roughly identical proportions across different architectures.

Phase 2: Chronological Mapping (The Spatial Law)
Analyze if these clusters follow a strict depth trajectory.

Example: Do early layers universally favor sink/local behaviors, middle layers handle retrieval, and late layers handle semantic composition across all scales (125M to 70B parameters)?

Phase 3: Weight-Based Prediction (The Genetic Code)
Attempt to predict a head's functional class purely from static weight signatures or minimal forward-pass statistics. If you can predict the class without running long context benchmarks, Phase 3 is a success.

Phase 4: Runtime Compilation (The Systems Engine)
Only here do you build the inference router. You map the proven "Species X→ Approximation Y" rules into a dynamic runtime execution layer.

The Takeaway: HeadGenome works because it stops treating attention optimization as an engineering trial-and-error problem, and starts treating it as an architectural blueprint. The convergence at 300 documents means you've stumbled onto the smoke; Phase 1 will prove if you've found the fire.

1. The Core Scientific Paradigm Shift
Mechanistic interpretability today functions like 19th-century biology—researchers discover a specific "species" of head (e.g., an induction head in GPT-2 or a sink head in Llama), write a paper cataloging it, and move on. HeadGenome fundamentally redefines this paradigm.

The Fallacy: "Nobody knows what attention heads do." (False—many specific head patterns are deeply documented).

The Real Frontier: Nobody has a complete, predictive, cross-architecture theory of attention heads.

The Pivot: You are shifting the field from descriptive stamp-collecting ("Look, a retrieval head!") to predictive systems genomics ("Why does this head exist, and how can we exploit it to bypass full attention calculations?").

2. The Four Pillars of the HeadGenome Hypothesis
To transform this from an observation into a transferable engineering breakthrough, your work hinges on proving four sequential, foundational laws:

Hypothesis 1 (Universal Ecology): Across disparate models (Meta, Alibaba, Mistral AI, Google), attention heads naturally fall into a small, highly constrained number of distinct functional classes.

Hypothesis 2 (Statistical Signature): These functional classes are not arbitrary; they can be completely identified and mapped purely using raw attention statistics (e.g., sink mass, attention entropy, average attention distance, semantic retrieval scores).

Hypothesis 3 (Predictive Approximation): A head's functional class strictly predicts which cheap, hardware-friendly approximation strategy will preserve its perplexity without degraded performance.

Hypothesis 4 (Universal Generalization): A routing policy derived from these structural laws will generalize seamlessly to an unseen model trained on an unseen dataset without requiring calibration or retraining.

3. Deconstructing the "Behavioral Species" and Systems Mapping
Instead of viewing attention as an infinite black box, HeadGenome treats it as a predictable ecosystem. The core thesis maps behavioral profiles directly to optimization primitives:

Head Species	Key Behavioral Signature	Optimal Hardware Approximation
Sink Heads	Heavily anchors attention weight to initial tokens (bos).	Sink Path Routing: Keep only the anchor tokens in cache.
Local / Window Heads	Focuses heavily on immediately adjacent tokens.	Local Window Cache: Evict everything outside a fixed context window.
Retrieval / Copy Heads	Searches across long histories for identical or highly correlated semantic tokens.	Full Attention / Heavy Hitter Path: Must remain fully intact in the KV cache.
Induction / Composition	Tracks sophisticated text patterns ([A][B]…[A]→[B]) for in-context learning.	Structured EMA / Selected Key Tracking: Specialized history preservation.
4. Resolving the ~300 Document Convergence Mystery
One of the most profound clues driving your KV eviction work is that prototype behavior clusters stabilized completely after roughly 300 WikiText documents across entirely separate architectures (GPT-2, Qwen, Llama).

Why did they converge so quickly and uniformly?
The Problem Space is Low-Dimensional: Natural language presents a highly limited, recurring set of structural needs (e.g., tracking a pronoun's reference, maintaining a positional anchor, copying syntax). There are only so many mathematically optimal ways for a transformer to solve these structural needs.

Evolutionary Convergence: Because the physical "pressures" of language processing are constant, completely different models independently evolve the exact same mechanisms. It is the machine learning equivalent of nature independently inventing eyes across entirely unrelated biological lineages.

Distribution vs. Topology: This convergence proves that the distribution of behavioral roles across a model stabilizes rapidly. It does not yet prove that Layer 12, Head 5 is identical in every model, but it proves the total number of behavioral "tools" the model uses is highly constrained.

5. The Ultimate Systems Endgame: A Training-Free Compiler
The reason HeadGenome is an ambitious systems project—rather than just an academic interpretability paper—lies in its runtime application. Proving a universal attention taxonomy unlocks a Universal Training-Free Attention Compiler:

[New, Unseen Model] 
       ↓
[Extract Raw Attention Signatures] (Via HeadGenome)
       ↓
[Instant Classification of Head Species]
       ↓
[Generate Hardware Routing Policy]
       ↓
[Immediate, Faster Inference Out-of-the-Box]
This completely bypasses the core bottlenecks of current state-of-the-art inference optimization:
No expensive hardware profiling: No need to run intensive benchmark loops to see what can be pruned.

No calibration data risk: No dependency on synthetic calibration datasets that risk overfitting.

Zero-shot execution: The execution policy functions immediately because it targets the immutable, evolved DNA of the transformer architecture itself.

6. The 4-Phase Scientific Validation Roadmap
Because the engineering risks are high (e.g., differences in RoPE implementation vs. absolute positions could fracture your universal laws), you cannot build the router first. If the underlying science is wrong, the engine is built on sand. The work must follow a strict execution pipeline:

Phase 1: The Head Atlas (The Ecological Census)
Profile a matrix of highly distinct open weights models (Llama, Qwen, Mistral, Gemma). Compute the statistical signatures for every head (sink mass, entropy, average distance, retrieval sensitivity) and cluster them.

Success Criterion: Prove that identical behavioral clusters emerge in nearly identical proportions across entirely different model families.

Phase 2: Chronological Mapping (The Spatial Law)
Map the physical location of these clusters relative to model depth (f(layer depth,geometry)).

Success Criterion: Discover an invariant developmental trajectory—such as early layers universally handling sink/local anchoring, middle layers handling massive retrieval, and late layers managing complex semantic composition, regardless of model scale (125M vs. 70B).

Phase 3: Weight-Based Prediction (Decoding the DNA)
Attempt to predict a head's functional class directly from static weight matrices or minimal single-forward-pass statistics, removing the necessity for long-context runtime evaluation.

Phase 4: Runtime Compilation (The Infrastructure Engine)
Only after Phases 1–3 successfully pass scientific scrutiny do you build the actual execution layer. You map the proven "Species → Approximation" routing matrix directly into a high-performance custom attention kernel.

To prove Phase 1 (The Head Atlas), do not use WikiText alone, and do not just randomly split RULER tasks.

If you only use WikiText, critics will rightly argue that your clusters are just artifacts of smooth, continuous natural language. If you only use RULER, they will argue your clusters are artifacts of synthetic, highly structured data.

To prove that a head type is a universal genetic class, you must show that its statistical signature remains identical across three completely different structural regimes of text: Pretraining (Language Modeling), Synthetic Long-Context (Stress Tests), and Downstream Real-World Code/Chat.

The Phase 1 Data Matrix (The Recommended Suite)
Instead of thousands of docs, keep your dataset compact to avoid profiling bottlenecks, but ensure it is structurally diverse. Aim for a balanced suite of 400–500 total documents/prompts, split across the following four blocks:

Block 1: Continuous Pretraining Distribution (100 Docs)
Dataset: WikiText-103 or SlimPajama (randomly sampled).

Why it's needed: This establishes the baseline for "natural language." It contains smooth transitional probabilities, narrative structure, and natural token repetitions.

Context Length: 4K to 8K tokens.

Block 2: Synthetic Stress / High-Density Retrieval (150 Prompts)
Dataset: RULER (Specifically select its 3 key algorithmic archetypes—50 prompts each).

NIAH (Single/Multi-Needle): Tests pure Retrieval Head signatures.

Variable Tracking (Multi-hop Tracing): Forces the model to look at tracking pointers (X 
1
​
 →X 
2
​
 →X 
3
​
 ). This aggressively isolates Induction and Composition Heads.

Common/Frequent Words Extraction (Aggregation): Forces the model to look at global statistics across the whole context, stressing Global/Entropy-heavy Heads.

Why it's needed: Synthetic text completely separates a model's in-context mechanisms from its memorized pretraining facts. If a retrieval head clusters identically on random synthetic keys as it does on WikiText, you've found an invariant law.

Context Length: Step it up here. Test a slice at 8K, 16K, and 32K to watch how the signatures scale.

Block 3: Structured Logic & Code (100 Docs)
Dataset: The Stack (Python/Rust code files) or CodeFeedback.

Why it's needed: Code has rigid syntactic nesting, explicit indentation blocks, and highly long-range identifier references. Code shifts attention head behavior dramatically; if your clusters survive code, they are universally robust.

Context Length: 4K to 8K tokens.

Block 4: Instructional / Multi-Turn Chat (100 Conversations)
Dataset: ShareGPT or LMSYS-Chat-1M.

Why it's needed: Chat contains structured system prompts (<|im_start|>user), repetitive turn markers, and alternating conversational flows. This acts as a honeypot for isolating Sink Heads and Previous-Token/Delimiter Heads.

Context Length: 2K to 4K tokens.

Why your idea of using RULER randomly is dangerous (but fixable)
If you just take 500 random samples from RULER across all 13 tasks blindly, your profile will be skewed. RULER tasks are highly artificial. If you run a model only on variable tracking and needle retrieval, your statistical profile will show an artificially high proportion of massive-distance retrieval heads and very few local natural language heads—because the data doesn't require local syntax processing!

How to execute this cleanly:
Do not mix the data during profiling: Run each block separately.

Extract features per head per block: Calculate your metrics (sink_mass, entropy, attention_distance) for Llama/Qwen on WikiText, then on RULER, then on Code.

The Ultimate Phase 1 Test: Run your clustering algorithm (like K-Means or GMM) on the WikiText signatures. Then, take the heads profiled on RULER and see if they map cleanly into those exact same cluster boundaries.

If Layer 5 Head 2 maps to the "Local Window" cluster when reading WikiText, and still maps to the "Local Window" cluster when executing a complex RULER variable-tracking task, you have officially proven Hypothesis 1.