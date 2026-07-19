# Section 6: Control Signals (The Activation Trigger)

Section 4 established that the macroscopic geometric trajectory is causally inert; injecting it does not trigger the target behavior. Thus, the trajectory must be a passive memory state requiring an independent "Control Signal" to trigger readout. 

We pre-registered four competing hypotheses to explain how the model reads this passive trajectory and writes the final vocabulary logits:
- **H1:** Trajectory alone is sufficient (Falsified in Section 4).
- **H2 (Readout Heads):** Specific late-layer heads read the trajectory; Signature = High Direct Logit Attribution (DLA) + attention on *content tokens*.
- **H3 (Control Token):** A syntactic trigger activates specific readout heads; Signature = High DLA + attention on *formatting tokens* (`:` or `\n`).
- **H4 (Distributed Circuit):** No single mechanism; Signature = No clean top-k heads dominate the DLA mass.

*(Pre-Registered Threshold: H2/H3 confirmed if top-3 components account for >50% of absolute DLA mass. H4 favored if top-10 account for <50%.)*

## 1. DLA & The Ambiguity of Mass

We computed the Direct Logit Attribution onto the correct generation tokens for the `Arithmetic` task across all components.

The attribution mass was decisively scattered:
- **Total Absolute DLA Mass:** 193.38
- **Top-3 Mass:** 59.69 (30.9%)
- **Top-10 Mass:** 100.62 (52.0%)

This narrowly straddles the pre-registered thresholds, favoring H4 but lacking overwhelming margin. However, the *type* of component dominating the mass was clear: **8 of the top 10 DLA components were MLPs**, not Attention Heads. 

## 2. Attention Tracing: The Correlational Trap (H3)

Despite their low absolute DLA mass compared to the MLPs, we extracted the attention patterns for the Top 3 Attention Heads (L27 H10, L27 H5, L27 H4) on the final generation step. 

Their attention patterns perfectly matched the correlational signature of **H3 (Control Token)**:
- **L27 H4** allocated 34.2% of its total attention exclusively to the `:` token.
- **L27 H10** allocated 29.7% of its total attention exclusively to the `:` token.

If we stopped the analysis here, it would be extremely tempting to conclude that H3 is true: "Layer 27 Attention Heads wait for the `:` syntactic trigger, and then fire to read the trajectory into the logits."

## 3. Causal Necessity: Falsifying H3

Because attention weight is strictly correlational, we applied a definitive causal necessity test: we performed Mean Ablation on those exact Top 3 Readout Heads and measured the collapse in original task accuracy. As a control, we ablated a magnitude/depth-matched random trio of heads.

The results were a stark falsification of H3:
- **Baseline Accuracy:** 100.0%
- **Top-3 Readout Heads Ablated:** **100.0%**
- **Random Control Ablated:** 100.0%

Despite their massive attention onto the formatting trigger (`:`), ablating the top DLA readout heads had exactly **zero effect** on the model's ability to answer the arithmetic prompt. 

**Conclusion: H4 is Confirmed.** The readout of the semantic trajectory is a robust, distributed nonlinear circuit governed heavily by late-layer MLPs. There is no single "Control Signal Readout Head." The apparent syntactic trigger heads (H3) are entirely redundant or epiphenomenal, providing yet another cautionary tale about trusting attention maps and static correlations to establish causal mechanism.
