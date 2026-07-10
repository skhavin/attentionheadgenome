# FDR Correction Ledger (Benjamini-Hochberg)

This ledger tracks all phase-level statistical tests to enforce the Global False Discovery Rate (FDR) correction pre-registered in `plan.md`.

*Target FDR Control ($q$)*: 0.05
*Total Pre-Registered Hypotheses (Estimated)*: ~36 (12 phases $\times$ 3 models, though some phases are descriptive).

## Phase Results Tracker

| Phase | Model | Test Type | Raw $p$-value | Effect Size (Magnitude) | Status (Raw) | Status (FDR Corrected) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Qwen2.5-1.5B | Wilcoxon (Discovery, N=40) | $1.75 \times 10^{-8}$ | Cliff's $\delta = -0.93$ (Ceiling 87.5%) | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 1** | Qwen2.5-1.5B | Wilcoxon (Confirmation, N=20) | $4.94 \times 10^{-4}$ | $\Delta_{median} \approx 4$ layers | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 2** | Qwen2.5-1.5B | Paired t-test vs Uniform (Conf, Pooled N=35) | $4.25 \times 10^{-7}$ | Cliff's $\delta = 0.75$ | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 2** | Qwen2.5-1.5B | Paired t-test vs Positional (Conf, Pooled N=35) | $1.14 \times 10^{-12}$ | Cliff's $\delta = 0.99$ | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 3** | Qwen2.5-1.5B | Wilcoxon (True vs Placebo, L22, N=20) | $2.85 \times 10^{-1}$ | $\Delta = -7.7\%$ restoration | <span style="color:gray">Null Result</span> | *Pending Global End* |
| **Phase 3** | Qwen2.5-1.5B | Wilcoxon (True vs Placebo, L25, N=20) | $6.44 \times 10^{-2}$ | $\Delta = +4.3\%$ restoration | <span style="color:gray">Null Result</span> | *Pending Global End* |
| **Phase 3** | Qwen2.5-1.5B | Wilcoxon (True vs Placebo, L27, N=20) | $8.08 \times 10^{-1}$ | $\Delta = +30.8\%$ restoration | <span style="color:gray">Null Result</span> | *Pending Global End* |
| **Phase 4** | Qwen2.5-1.5B | Wilcoxon (Retrieval DLA > Random DLA, N=20) | $4.04 \times 10^{-1}$ | $\Delta_{mean} = -0.22$ logit | <span style="color:gray">Null Result</span> | *Pending Global End* |
| **Phase 4** | Qwen2.5-1.5B | Wilcoxon (Late MLP DLA > Retrieval Head DLA, N=20) | $1.33 \times 10^{-5}$ | $\Delta_{mean} = +6.60$ logit | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 6/7** | Qwen2.5-1.5B | Wilcoxon (Top 5 A2A Edges vs Placebo, N=20) | $min(p) = 0.187$ | Cliff's $\delta_{mean} \approx 0.07$ | <span style="color:gray">Underpowered (Mean Power 40%)</span> | *Pending Global End* |
| **Phase 8** | Qwen2.5-1.5B | Mann-Whitney U (Boost-Correct vs Neutral Causal Drop, N=20) | N/A (Structurally Unconfirmable) | $\Delta \approx 0.03$ logit | <span style="color:gray">Underpowered (Power 9.8%)</span> | *Pending Global End* |
