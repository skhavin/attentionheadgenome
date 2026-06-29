<div align="center">
  <h1>🧬 The HeadGenome Project</h1>
  <p><strong>A Structural and Behavioral Taxonomy of Attention Heads in Large Language Models</strong></p>
  
  <p>
    <a href="https://github.com/skhavin/attentionheadgenome/issues"><img alt="Issues" src="https://img.shields.io/github/issues/skhavin/attentionheadgenome?color=1e293b&style=flat-square"></a>
    <a href="https://github.com/skhavin/attentionheadgenome/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/skhavin/attentionheadgenome?color=1e293b&style=flat-square"></a>
  </p>
</div>

---

The **HeadGenome Project** is a comprehensive mechanistic interpretability framework that maps the exact functional ecology of attention heads across modern transformer architectures (GPT-2, Qwen-2.5, and Llama-3.2). 

By analyzing over 1,500 attention heads, this repository demonstrates that transformer attention mechanisms are not chaotic, homogeneous systems. Instead, they follow a rigorous **spatial scaling law**, evolving from low-entropy structural sinks into highly specialized semantic **Retrieval** and **Induction** sub-species as they progress through the network depth.

## 📖 Key Findings

1. **The V/Q Spatial Scaling Law:** The functional role of an attention head is strictly governed by its $||W_V|| / ||W_Q||$ norm ratio, dictating a mathematically inevitable spatial progression.
2. **Circuit Co-Gating:** Semantic retrieval heads and structural induction heads cannot function independently; they form a co-dependent pipeline.
3. **Data Independence (The Permutation Null):** The specialized topology of the HeadGenome is not a byproduct of learning English language statistics. It is an architecture-intrinsic geometric necessity that survives total semantic destruction (shuffled tokens) and applies equally across massively divergent training domains (English, Code, Math).
4. **The Perplexity Illusion:** A model can achieve near-perfect cross-entropy loss (perplexity) even when its long-range structural routing (Needle-In-A-Haystack retrieval) has completely collapsed.

## 🗂 Repository Structure

This repository has been carefully structured for research reproducibility and navigation:

```text
attentionheadgenome/
├── consolidated_research_report.md       # The consolidated technical summary of all findings
├── README.md                             # You are here
├── lib/
│   └── headgenome/                       # Core analytical framework, sparse routing masks, and evaluation benchmarks
├── scripts/                              # Experimental scripts and visualization tools
│   ├── phase10_vq_universality.py        # Initialization Null experiments
│   ├── phase11_permutation_null.py       # Permutation Null (shuffled sequence) stress-tests
│   ├── phase11_cross_domain_proof.py     # Cross-Domain V/Q structural invariance proof
│   ├── generate_phase9_figures.py        # Generates the lexical anatomy visualisations
│   └── ...                               # (All other phase scripts used in the study)
├── docs/                                 # Historical research plans, phase documentation, and field notes
└── outputs/                              
    ├── final_artifacts/                  # Contains the final HeadGenome_Master_Report.md
    ├── phase9_semantics/                 # Token-level lexical profiling artifacts (Figure 7)
    ├── phase10_universality/             # V/Q Initialization Null plots (Figure 8)
    ├── phase11_permutation_null/         # Gibberish stress-test plots (Figure 9)
    └── phase11_universality/             # Cross-Domain structural proof (Figure 10)
```

## 📊 Major Visual Artifacts

The core arguments of the HeadGenome Project are proven in the following generated figures:

### 1. The HeadGenome Atlas
A comprehensive architectural map showing exactly where functional head types emerge across network depth and across different model scales.
<p align="center">
  <img src="outputs/final_artifacts/headgenome_atlas.png" width="100%">
</p>

### 2. The Entropy-Collapse Scatterplot
Proving the clear functional clustering of attention heads based on their behavior across structural and semantic stress tests.
<p align="center">
  <img src="outputs/final_artifacts/headgenome_map.png" width="80%">
</p>

### 3. The Lexical Anatomy (Figure 7)
Demonstrates the massive difference in natural-language token distributions preferred by each of the four Head categories.
<p align="center">
  <img src="outputs/phase9_semantics/figure7_lexical_anatomy.png" width="100%">
</p>

### 4. The Initialization Null / Training Emergence (Figure 8)
Proves that the V/Q scaling law is absent in randomly initialized (untrained) networks, establishing it as an emergent property of gradient descent.
<p align="center">
  <img src="outputs/phase10_universality/figure8_vq_emergence.png" width="100%">
</p>

### 5. The Permutation Null (Figure 9)
Shows that when models are fed perfectly shuffled gibberish tokens (destroying syntax and meaning), Induction heads actually *strengthen* their structural firing, while Retrieval heads attenuate, proving functional specialization is structurally hardcoded, not semantically guessed.
<p align="center">
  <img src="outputs/phase11_permutation_null/figure9_permutation_null.png" width="100%">
</p>

### 6. The Cross-Domain Proof (Figure 10)
Maps the V/Q scaling Pearson correlation across WebText (GPT-2), Multilingual/Code (Qwen), and Multilingual/Math (Llama) to definitively prove that the taxonomy is entirely data-agnostic.
<p align="center">
  <img src="outputs/phase11_universality/figure10_cross_domain.png" width="100%">
</p>

## 🚀 Getting Started

To reproduce the analysis or run the `headgenome` routing policies:

1. **Install Requirements:**
   ```bash
   pip install torch transformers datasets matplotlib numpy scipy tqdm
   ```

2. **Run Core Experiments:**
   All scripts are housed in the `scripts/` directory. Run them from the repository root:
   ```bash
   # Example: Run the Permutation Null experiment
   python scripts/phase11_permutation_null.py
   ```

## 📜 Full Documentation
For a complete theoretical and empirical breakdown of the methodology, ablation studies, and mathematical formalism, please refer to the [**HeadGenome Master Report**](outputs/final_artifacts/HeadGenome_Master_Report.md) or the [**Consolidated Research Report**](consolidated_research_report.md).
