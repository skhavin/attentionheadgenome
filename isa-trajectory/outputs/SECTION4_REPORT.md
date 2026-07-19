# Section 4 Report: Causal Intervention & Steering

Sections 1-3 established that models construct computation via universally conserved, progressive geometric expansions (the "Trunk and Branch" manifold). A natural, compelling hypothesis is that this macroscopic geometric structure is the actual causal mechanism driving the model's behavior—that the causal power of a trajectory perfectly tracks its geometric maturation (the F-Ratio curve).

To test this **Sufficiency** hypothesis, we employed Sliding-Window Contrastive Activation Addition (CAA). We injected mature operational trajectory vectors into the residual stream at the generation token across all layers of Qwen2.5-1.5B, testing if we could hijack the model's behavior.

## 1. Rigorous Methodology
Given the history of satisfying-but-fragile causal stories in interpretability, we pre-registered an extremely strict methodology:
1. **The 3-Bucket Classifier**: Output was deterministically categorized via Regex into (A) Source/Correct, (B) Genuine Target Hijack (e.g., must contain explicit comma-separated lists or fixed comparative vocabulary), and (C) Garbage/Neither.
2. **Magnitude-Matched Random Control**: At *every* layer and multiplier $c \in \{1.5, 3.0, 5.0\}$, we injected the true steering vector and a randomly sampled Gaussian vector normalized to the exact same $L_2$ norm. 
3. **Exact Significance Testing**: A causal hijack is only considered successful if the True Vector's Bucket B rate significantly beats the Random Vector's Bucket B rate across 30 paired prompts (exact McNemar's $p < 0.05$). This definitively isolates targeted geometric steering from generalized disruption.
4. **Pre-Registered Pass Bar**: The causal hijack success rate must strongly correlate ($r > 0.7$) with the F-Ratio maturation curve.

## 2. Findings: The Honest Null Result

The strict methodology yielded a massive, clean falsification of the macroscopic causal hypothesis. The geometric manifold exists, but it is **epiphenomenal** to continuous causal generation.

### Hypothesis Falsified: Correlation is Near Zero
The hypothesis that causal efficacy tracks the geometric maturation F-Ratio is completely falsified. The full-curve Pearson correlations are near zero:
- `Arithmetic -> Sorting`: **$r = 0.049$**
- `Fact Recall -> Comparison`: **$r = 0.112$**

Even restricting the correlation exclusively to the "Rise Phase" (Layers 0 to 21, ignoring post-peak dynamics) yields zero correlation ($r = 0.019$ and $r = 0.161$).

![Causal vs F-Ratio](causal_intervention/causal_vs_fratio.png)

### Causal Efficacy is a "Sparse Flash", not a Smooth Manifold
Instead of a smooth causal gradient tracing the geometric divergence, the intervention success is $0\%$ across most of the network, punctuated by massive, highly localized "flashes" of vulnerability.
- For `Fact Recall -> Comparison`, the hijack rate sits at $0\%$ until **Layer 11**, where it suddenly spikes to **$83\%$** (Real $B=0.83$, Random $B=0.00$, $p=0.0000$). By Layer 13, it instantly collapses back to $0\%$.
- For `Arithmetic -> Sorting`, it flashes briefly at **Layer 17**, while remaining completely robust everywhere else in the mid-network.

Because we ran the magnitude-matched random control, we know these flashes are *real* target-specific steering—the random control produced 0% hijack at those same layers. But this causal power is entirely divorced from the macroscopic, continuous geometric expansion.

### The "End-of-Network" Logit Injection Effect
As predicted by our third pre-registered outcome mode, hijack success spikes massively again at the very end of the network for both pairs (Layers 25-27), right before generation. At Layer 27, `Fact->Comp` hits **$73\%$** hijack success, and `Arith->Sort` hits **$43\%$** success. Injecting the target centroid right before the Language Modeling head bypasses the internal trajectory maturation entirely and acts as a direct injection into the output vocabulary logits.

---

## 3. Conclusion

This section perfectly extends the "honest null" story of Paper 1. Sections 1-3 prove that models *do* possess universally conserved geometric trajectories for cognitive operations. But Section 4 proves that you cannot simply "push" the network along this manifold anywhere to hijack it. 

The model's actual causal mechanism for generation is **sparsely gated** (likely by specific attention heads firing at L11 or L17), completely ignoring the smooth, 20-layer geometric expansion happening in the residual stream. The macroscopic geometry mathematically diverges, but causal power is localized, not continuously distributed across the manifold's depth.

**Artifacts:**
- **Script**: `code/step4_causal_intervention.py`, `code/step4_plot_results.py`
- **Data**: `outputs/causal_intervention/sweep_results.json`
- **Plots**: `outputs/causal_intervention/causal_vs_fratio.png`
