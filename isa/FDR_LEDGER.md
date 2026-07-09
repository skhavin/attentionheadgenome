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
| **Phase 3** | Qwen2.5-1.5B | Wilcoxon (True vs Placebo, L23, N=7) | $7.81 \times 10^{-3}$ | $\Delta = +25.0\%$ restoration | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 3** | Qwen2.5-1.5B | Wilcoxon (True vs Placebo, L22, N=7) | $7.81 \times 10^{-3}$ | $\Delta = +21.0\%$ restoration | <span style="color:green">Significant</span> | *Pending Global End* |
| **Phase 3** | Qwen2.5-1.5B | Wilcoxon (True vs Placebo, L24, N=7) | $1.56 \times 10^{-2}$ | $\Delta = +11.1\%$ restoration | <span style="color:green">Significant</span> | *Pending Global End* |
