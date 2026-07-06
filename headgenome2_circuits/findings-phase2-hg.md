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
*Robustness Note: We initially ran this on $N=26$ pairs, yielding $p=0.0425$. To ensure the effect size was stable and not a fragile artifact of small N, we expanded the dataset and re-ran the test across $N=50$ causal intervention pairs.*

The expanded causal patch worked perfectly:
* **Null Shift:** Injecting the `Count=X+2` state from Null Heads shifted the output by an average of `+1.38`.
* **Counting Head Shift:** Injecting the `Count=X+2` state from the identified Counting Heads shifted the output by an average of **`+5.68`**.
* **Wilcoxon p-value:** `0.0000` (highly significant).

**Falsification Passed: `TRUE`.** 

The identified Counting Heads are definitively, causally responsible for transmitting the list count.

---

## Falsification of the Structured Output Circuit (JSON Delimiters)

Our third probe tested the hypothesis that specific "Delimiter Heads" are responsible for tracking structured syntax boundaries (like nesting braces `{}` or `[]`). We hypothesized that ablating them would cause syntax collapse (e.g., unbalanced brackets).

### The Observational Illusion (Again)
We probed Qwen2.5-0.5B on a dataset of nested JSON objects requiring closing braces (`}}}`). 
The probe structurally isolated specific heads (e.g., `L11H13, L9H7`) that allocated an astonishing **$1.000$ (100%)** of their attention mass directly onto the open braces (`{`) during the generation of the closing braces.

This was the most extreme structural specialization we had seen yet. 

### The Causal Reality & The Ceiling Effect
We subjected these heads to the exact same rigorous causal ablation trap used for the Copy Circuit (comparing targeted ablation against depth/entropy-matched null ablation on the metric of JSON bracket validity).

Crucially, to verify the ablation worked mechanically, we integrated an strict execution-time assertion hook (`assert torch.all(x[...] == 0.0)`) into the `o_proj` tensor. The assertion passed, confirming the heads were truly obliterated.

Furthermore, to avoid a "ceiling effect" (where the task is too easy for the model to fail), we radically increased the dataset difficulty, injecting 5-7 levels of deep object nesting alongside array distractors.

The results were completely staggering:
* **Baseline Validity:** 100.0%
* **Null Ablation Validity:** 100.0%
* **JSON Head Ablation Validity:** **100.0%**

**Falsification Passed: `FALSE`.**

Despite dedicating literally 100% of their attention to tracking the open brackets, destroying these heads caused **zero disruption** to the model's ability to successfully generate the closing brackets. 

*Critical Limitation:* Because the Baseline Validity remained at exactly 100.0% even with 7 levels of nesting, we encountered a hard ceiling effect. The JSON closure task is simply too trivial for Qwen2.5-0.5B. While it suggests extreme redundancy (the "Illusion of Structural Attention"), we cannot conclusively state there was *no* degradation because there was no performance room left to fall.

---

## Validation of Attention $\leftrightarrow$ MLP Routing (Circuit 4)

Our final phase investigated the structural and causal interface between Attention Heads and downstream MLPs. Specifically, do the weights of an Attention Head's output matrix ($W_O$) structurally align with the weights of a downstream MLP's input matrices to reliably predict causal routing?

> [!WARNING]
> **Scope Limitation:** This investigation was a single-circuit case study using exclusively the causally-confirmed Counting Heads (from Circuit 2) as the source. It evaluates routing for *one specific circuit type*. It is not yet a general test of the overarching Frobenius-routing hypothesis, and we do not generalize this single instance into a universal law of attention-MLP routing.

### The Structural Probe
We extracted the Counting Heads from Layer 16. To account for Qwen's SwiGLU non-linearity, we fused the MLP input gates by taking the element-wise Hadamard product of the gate and up projections: $W_{fused} = W_{gate} \cdot W_{up}$.
We then computed the Frobenius Norm of the interaction: $|| W_{fused} \cdot W_O ||_F$ for all downstream MLPs.

The structural probe identified **MLP 21** as having the overwhelmingly strongest parameter connectivity to the L16 Counting Heads (Fused Norm = $0.32$).

### The Empirical Causal Test
To verify if this structural connectivity governs actual causal routing, we implemented a targeted inter-layer causal patch (Count=X patched into Count=X+2). We measured the shift in the target MLP's pre-activation state ($\Delta MLP_{target}$).

To establish a strict empirical noise floor, we ran the exact same patch across 5 distinct Null Head groups. Critically, these null heads were strictly depth-matched: they were drawn exclusively from the same layer (Layer 16) as the Counting Heads. By matching depth, we guarantee that the null distribution is not artificially deflated by comparing against early-layer heads (which have vastly different output magnitudes), ensuring a fair and rigorous causal baseline.

The causal reality was striking:
* **Null Distribution (L16 Heads):** $\mu = 27.530$, $\sigma = 18.437$
* **Empirical 2-Sigma Threshold:** $64.404$
* **Target (Counting Head) Shift on MLP 21:** **$117.835$**

### The Statistical Correlation
The structural Frobenius Interaction Norms across the downstream MLPs predicted the measured causal activation shift with near-perfect linear correlation:
* **Pearson r:** $0.996$
* **p-value:** $0.0000$
* **Sample Size (N):** $7$ (Downstream MLPs L17-L23)

**Falsification Passed: `TRUE` (With Caution).**

While the targeted shift on MLP 21 definitively proves causal routing, the correlation of $r=0.996$ must be treated with caution. Because there are only 7 downstream layers to correlate across ($N=7$), this near-perfect linearity is statistically fragile; a single outlier could easily skew it. We confidently claim that the *top-ranked structural connection matches the top causal route*, but we temper the claim that parameter norms perfectly map to causal flow across *all* layers until tested on deeper architectures with a higher $N$.

For this specific semantic counting circuit, the structural Frobenius norm of the fused SwiGLU matrices flawlessly predicted the causal routing path. The Counting Heads explicitly and causally broadcast their integer accumulations directly into the deep semantic processing block of MLP 21.

---

## Falsification of the Arithmetic Circuit (Circuit 5)

As the final test of Phase 2, we investigated whether the causally-confirmed Counting Circuit generalized into a broader "Numeric Accumulation" mechanism, or if it was narrowly tied to list-parsing. 

We applied the strict falsification template to single-digit arithmetic (addition):
`Question: What is X plus Y? Answer: The sum is [Z]`

### The Structural Probe
We probed Qwen2.5-0.5B to identify heads that allocated maximum attention mass to the two operand tokens (`X` and `Y`) during the generation of the sum. 
The probe identified a clear set of "Arithmetic Heads" allocating massive attention to the operands:
* **Top Heads:** `L17H8` (Mass: 0.699), `L14H13` (Mass: 0.656), `L16H8` (Mass: 0.572), `L16H7` (Mass: 0.564), `L16H11` (Mass: 0.521).

**Critical Observation:** The heads `L16H8`, `L16H7`, and `L16H11` are the *exact same heads* identified as the causally-active Counting Heads in Circuit 2. Structurally, it appeared the Counting Circuit was indeed generalizing to arithmetic computation by explicitly attending to mathematical operands.

### The Causal Reality
We subjected these Top Arithmetic Heads to the same rigorous inter-layer causal patch used previously, hooking the `o_proj` output and verifying the patch via runtime assertions. We extracted the expected value of the output digit over the softmax distribution to measure the shift toward the injected source sum.

The causal patch was tested across $N=50$ pairs, compared against strictly depth-matched Null Heads, and evaluated via a pre-registered Wilcoxon signed-rank test.

The results completely shattered the structural hypothesis:
* **Avg Null Shift towards Source:** $-0.023$
* **Avg Target Shift towards Source:** $-0.333$
* **Wilcoxon p-value:** `0.9838`

**Falsification Passed: `FALSE`.**

Despite dedicating nearly 70% of their attention mass to the mathematical operands, and despite being the exact same heads that causally govern list-counting, these heads had **absolutely zero causal impact** on the model's arithmetic output. 

This confirms two major findings:
1. The **Observational Illusion** strikes again. Attention to operands does not mean the head is performing or routing the arithmetic computation (which is likely handled entirely within the MLPs).
2. The Counting Circuit is **narrowly specialized** for sequential list-parsing. It does not generalize to mathematical addition, proving that structural overlap (the same heads firing) does not guarantee functional generalization.
