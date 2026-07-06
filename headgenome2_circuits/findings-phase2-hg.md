# HeadGenome II: Algorithmic Circuits
## Falsification of the Universal Copy Circuit

### 1. The Observational Illusion
During our structural probing phase, we hypothesized the existence of a "Universal Copy Circuit": a dedicated set of attention heads responsible for exact-match string copying (e.g. duplicating synthetic UUIDs or rare identifiers from context to output). 

When we probed **Qwen2.5-0.5B** on a synthetic UUID context dataset ($N=50$), the structural evidence was overwhelming. We identified four top "Copy Heads":
* **L13H13**
* **L9H13**
* **L13H11**
* **L13H9**

During generation, these heads routed **~75% to 80%** of their total attention mass *directly* onto the exact tokens of the target UUID in the context. In a standard visualization-first mechanistic interpretability study, this would definitively be classified as the "UUID Copying Circuit".

### 2. The Falsification Methodology
However, per the strict falsification-first discipline established in this project, high attention mass does not imply causal necessity. We pre-registered the following failure criteria:
1. **The Null Baseline:** Ablating the identified "Copy Heads" must cause a significantly larger drop in performance than ablating a set of Random Heads.
2. **Strict Matching:** The Null Heads cannot be chosen purely at random. They must be strictly matched to the targeted Copy Heads on both **layer depth** (to control for phase confounds) and **baseline attention entropy** (to ensure we aren't comparing a highly focused head to a lazy/diffuse head). 

To achieve this, we first ran a baseline entropy profiler over 50 continuous chunks of Wikitext-2 to calculate the baseline attention entropy for every head in the model. We then identified 4 Null Heads `[(13, 12), (9, 11), (13, 0), (13, 6)]` that perfectly matched the layer and baseline entropy profile of the 4 Copy Heads.

### 3. The Causal Reality (Results)
We zeroed out the `o_proj` slice for the targeted heads and re-ran exact-match UUID generation across the 50-prompt dataset. To guarantee no silent hook failures, we embedded an explicit `assert torch.all(x[...] == 0.0)` sanity check directly into the PyTorch pre-hook during generation. 

The results completely falsified the hypothesis:
* **Baseline Accuracy:** 48.0%
* **Null Ablation Accuracy:** 50.0% *(Note: The +2% shift is negligible, representing a 1-prompt variance over N=50, and confirms no major capacity degradation).*
* **Copy Head Ablation Accuracy:** **48.0%**

**Falsification Passed: `FALSE`.**

Despite dedicating 80% of their computational bandwidth to routing exact-match tokens, destroying these heads had **absolutely zero effect** on the model's actual ability to copy the UUID. The circuit is fully redundant or entirely non-causal for the final generation path.

### 4. Cross-Model Pre-Registration
We are currently executing this exact rigorous ablation suite across four diverse architectures: **Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B, and Gemma-2B**.

**Quantitative Pre-registered Success Criteria for the Sweep:**
* **Individual Replication Bound:** A model replicates the "Observational Illusion" (redundancy) if the exact-match accuracy drop of the targeted Copy Ablation is $\le 10\%$ worse than the Null Ablation baseline (i.e., `acc_null - acc_copy <= 0.10`). If it drops $> 10\%$ more than the null, the circuit is considered causally validated.
* **The Sweep Threshold:** If $> 3/4$ models replicate the observational illusion boundary.
* **The Claim Limit:** We explicitly constrain the conclusion to avoid overreaching. If the threshold is met, the accurate claim is: *"Structural attention allocation is not a reliable indicator of causal necessity for exact-match string copying, at least across the model scale range tested."* We will not generalize this as a blanket "universal law of interpretability."

---

## Validation of the Counting Circuit

Unlike the Copy Circuit which failed causal testing, the **Counting Circuit** passed its strict pre-registered falsification trap on Qwen2.5-0.5B.

### The Inter-Layer Patch Test
We probed for heads that allocate high attention to list-item numbers (e.g. `1. Apple\n2. Banana\n3. Cherry`). We identified four specific counting heads primarily clustered at the very end of the network (`L16H11, L16H8, L16H7`). 

Instead of ablating them, we ran an inter-layer causal patch:
1. We cached the activation of the Counting Heads on a "Count=5" prompt.
2. We forcefully injected that cached activation into the Counting Heads during the generation step of a "Count=3" prompt. 
3. **Null Baseline:** We did the exact same injection, but using depth-matched random heads instead of the hypothesized counting heads.

### The Causal Reality (Results)
We pre-registered a **Wilcoxon signed-rank test** ($p < 0.05$) to mathematically prove that the target injection shifts the model's integer output more than the null injection.

The causal patch worked perfectly:
* **Null Shift:** Injecting the `Count=5` state from Null Heads shifted the output by an average of `+0.692`.
* **Counting Head Shift:** Injecting the `Count=5` state from the identified Counting Heads shifted the output by an average of **`+1.962`** (an almost perfect `+2` shift!).
* **Wilcoxon p-value:** `0.0425`

**Falsification Passed: `TRUE`.** 

The identified Counting Heads are causally responsible for accumulating and transmitting the list count to the final layers.

---

## Falsification of the Structured Output Circuit (JSON Delimiters)

Our third probe tested the hypothesis that specific "Delimiter Heads" are responsible for tracking structured syntax boundaries (like nesting braces `{}` or `[]`). We hypothesized that ablating them would cause syntax collapse (e.g., unbalanced brackets).

### The Observational Illusion (Again)
We probed Qwen2.5-0.5B on a dataset of nested JSON objects requiring closing braces (`}}}`). 
The probe structurally isolated specific heads (e.g., `L11H13, L11H11, L9H7`) that allocated an astonishing **$1.000$ (100%)** of their attention mass directly onto the open braces (`{`) during the generation of the closing braces.

This was the most extreme structural specialization we had seen yet. 

### The Causal Reality
We subjected these heads to the exact same rigorous causal ablation trap used for the Copy Circuit (comparing targeted ablation against depth/entropy-matched null ablation on the metric of JSON bracket validity).

The results were completely staggering:
* **Baseline Validity:** 100.0%
* **Null Ablation Validity:** 100.0%
* **JSON Head Ablation Validity:** **100.0%**

**Falsification Passed: `FALSE`.**

Despite dedicating literally 100% of their attention to tracking the open brackets, destroying these heads caused **zero disruption** to the model's ability to successfully generate the closing brackets. The "Illusion of Structural Attention" phenomenon struck again, perfectly replicating the exact same redundant behavior we observed in the Copy Circuit.
