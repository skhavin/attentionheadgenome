# Phase 7: O(W) Routing & Scale-Up

## Overview
The goal of this phase is to train and evaluate a lightweight 4-path MoE router (`sink`, `local`, `recurrence`, `full`) that dynamically predicts which attention approximation path to take on a per-head basis, drastically reducing FLOPs without hurting PPL.

## Steps

### [x] 1. Implementation & Smoke Testing
- [x] Implement the 4-path MoE logic (`sink_path`, `local_path`, `recurrence_path`, `full_path`).
- [x] Implement lightweight O(W) features to avoid computing full attention matrices.
- [x] Run a 50-doc smoke test on Qwen 0.5B.
- [x] **Bug Fixes Discovered:** Fixed `NaN` entropy loss, fixed `Qwen2Attention` RoPE hooks by shifting to `o_proj`, fixed evaluation loop hangs due to 32K context chunk sizes, and fixed `recurrence_path` convolutional weights magnifying activations by >2x.

### [/] 2. GPT-2 Medium Full Validation
- [ ] Train GPT-2 Medium with the full 500/1000/200-document curriculum.
- [ ] Evaluate `hard_routing` perplexity (Goal: ΔPPL < 0.5).
- [ ] Analyze routing statistics to confirm pure specialists collapse cleanly and regime-switchers bail out dynamically.

### [ ] 3. Qwen 0.5B Contrast Evaluation
- [ ] Train Qwen 0.5B with the full 500/1000/200-document curriculum.
- [ ] Evaluate `hard_routing` perplexity and routing statistics.
- [ ] **Expected Result:** Qwen 0.5B will likely show lower specialization and fewer cheap paths compared to GPT-2, proving that the MoE conversion relies heavily on existing architectural specialization.

### [ ] 4. LLaMA-3.1-8B Scale-Up
- [ ] Train LLaMA-3.1-8B with the full 500/1000/200-document curriculum (4-bit NF4 base, BF16 routers).
- [ ] Evaluate `hard_routing` perplexity and routing statistics.
- [ ] Calculate the percentage of "cheap paths" utilized to estimate FLOPs savings.

### [ ] 5. Paper Publication Narrative
- [ ] Synthesize results into the final paper.
- [ ] Highlight the Qwen failure/success contrast as a negative result data point that strengthens the paper.
- [ ] Document final routing statistics and layer-by-layer taxonomies.
