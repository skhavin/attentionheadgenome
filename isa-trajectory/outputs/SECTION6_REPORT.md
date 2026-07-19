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

**Conclusion:** The DLA mass distribution alone was ambiguous (straddling our pre-registered threshold), but the causal ablation result decisively rules out H2/H3 for the specific heads tested. By elimination, combined with the heavily MLP-dominated DLA mass, the evidence most strongly supports **H4 (Distributed Circuit)**. 

## 4. The Final Trap: MLP Necessity & Generic Fragility

If H4 is true and MLPs form a distributed readout circuit, we should be able to ablate the top-DLA MLPs and observe a targeted drop in accuracy. To avoid the 100% ceiling effect of our original validation set, we ran this necessity test on 80 novel Out-of-Distribution (OOD) arithmetic prompts. We pre-registered that a true distributed circuit should show *graceful degradation* (a small drop for individual ablations, and a large drop for group ablations).

**Results (OOD Arithmetic Test):**
- **Baseline Accuracy:** 93.8%
- **Top-1 MLP (L27) Ablated:** 86.2% *(Small drop, as expected)*
- **Top-3 MLPs Group Ablated:** 0.0% *(Massive drop, as expected)*
- **Random Control (3 MLPs) Ablated:** **0.0%**

**The Methodological Reality Check:** The Top-3 MLP ablation successfully destroyed the model's ability to do arithmetic. However, the *Random Control* (ablating 3 depth-matched MLPs: L21, L23, L24) *also* completely destroyed the model's accuracy. 

Because the random control failed, we **cannot claim** that the Top-3 DLA MLPs form the specific, unique causal circuit for arithmetic readout. Instead, this proves that the network's late-stage MLPs are generically fragile to mean ablation; knocking out any three of them causes total generation collapse. 

Once again, rigorous controls saved us from a false positive. We conclude that while steerable trajectories are built by MLPs (Section 5) and readout is distributed (Section 6), simple linear attribution (DLA/DTA) and mean-ablation are fundamentally insufficient to cleanly isolate the true, non-redundant causal subgraph of a dense model's late layers.
