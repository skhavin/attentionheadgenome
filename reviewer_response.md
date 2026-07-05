# Response to Reviewer Notes

## 1. Entropy Delta ($\Delta$) Sign and Focus

**Reviewer Note:** *"...H_task - H_base should give the signs opposite to what you see, so either there's a glossing error, or you intentionally inverted it."*

**Response:** You are correct that the formula as written in the report ($H_{task} - H_{baseline}$) contradicts the signs of the reported data. **Eq. 1 as printed has a sign error; the implementation and all reported results use $\Delta = H_{baseline} - H_{task}$, as intended.**

By tracking the exact variables in the extraction pipeline (`phase1/step12_robust_entropy_50pairs.py`):
1. `ctx_m` (Match Context, e.g., *"The capital of France is Paris."*) + `query` (*" The capital of France is"*) is passed to calculate `match_entropy` ($H_{task}$).
2. `ctx_nm` (Nonmatch Context, e.g., *"The weather today is sunny."*) + `query` is passed to calculate `nonmatch_entropy` ($H_{baseline}$).
3. Line 295 explicitly executes `delta = nme_val - me_val`.

Because the code subtracts task entropy from baseline entropy:
* A **positive** $\Delta$ means the head's entropy *decreased* during the task. This aligns perfectly with Retrieval heads shrinking their focus onto a specific needle token (the prior instance of "Paris").
* A **negative** $\Delta$ means the head's entropy *increased* relative to its baseline. Induction heads typically have very low baseline entropy (acting as highly focused local sliding windows). During an induction task, they split or broaden their attention across repeated structural tokens, causing a relative increase in entropy.

The data supports the hypothesis that retrieval heads shrink focus and induction heads broaden it relative to baseline; the discrepancy was solely a typo in how the $\Delta$ formula was documented in the report.

## 2. Threshold Selection (0.30 and -0.50)

**Reviewer Note:** *"There are 2 values in the entropy collapse right? Find me the mathematically best value for it"*

**Response:** The classification thresholds ($\Delta \ge 0.30$ for Retrieval, $\Delta \le -0.50$ for Induction) were not chosen blindly; they were derived post-hoc from the threshold sensitivity analysis to find where the data naturally separates. 

Looking at the sensitivity data across the three models (`outputs/phase1/threshold_sensitivity.json`), we tested retrieval thresholds from 0.15 to 0.45. The combined head drop-off rates across models are:
* 0.15 to 0.20: -19 heads
* 0.20 to 0.25: -19 heads
* 0.25 to 0.30: -13 heads
* **0.30 to 0.35: -4 heads**
* 0.35 to 0.40: -5 heads

The threshold of 0.30 was selected because it sits exactly where the distribution flattens out (the "elbow"). Thresholds below 0.30 capture a large, noisy gradient of Local heads, while increasing the threshold beyond 0.30 eliminates very few additional heads. We chose 0.30 post-hoc because it empirically isolates the stable cluster of specialized heads. 

The same methodology applies to the Induction threshold of -0.50, which sits at the point where the steepest drop-off in false positives stabilizes.

## 3. Head Extraction, Eviction, and "Microscope" Analysis

**Reviewer Note:** *"Which heads are you evicting? Are you extracting the heads in a general and global way? Because you should analyze the neurons and understand what those heads are doing under the 'microscope.'"*

**Response:**
* **Extraction:** The extraction is global. `canonical_classification.py` applies the empirically derived thresholds (0.30 and -0.50) to all heads across 50 diverse prompt pairs to generate a universal taxonomy.
* **Microscope Analysis:** To verify these macroscopic labels at the token level, `audit_head_vocabulary.py` was run over millions of WikiText tokens. It confirms that the heads classified as Retrieval disproportionately allocate attention mass to proper nouns and sentence starts, while Sink heads allocate mass strictly to punctuation and the BOS token. 
* **Eviction Policy:** In `phase4/step1_routing_policy.py`, KV cache eviction is applied dynamically during decoding based on these classifications. Local heads retain only a rolling window, while Retrieval and Induction heads retain full $O(N)$ cache access.

## 4. Dataset Profile

**Reviewer Note:** *"I need to know which dataset you're using, maybe with some general examples of the text."*

**Response:** The project uses two datasets for different phases:
1. **Entropy Probing:** A synthetic dataset of 50 prompt pairs across 5 semantic categories (e.g., Geography, Science). Example: Context: *"The capital of France is Paris."* / Query: *" The capital of France is"*.
2. **Perplexity / Eviction:** `Salesforce/wikitext-103-v1` (validation split) is used for long-context perplexity testing. Example: *" = = Valkyria Chronicles III = = \n Senjō no Valkyria 3..."*.

## 5. GQA Analysis and Layer Geometry

**Reviewer Note:** *"On the GQA, can you tell me which ablation test, and exactly how you classify and analyze the individual Q heads? How do you observe the internal geometry of the layers?"*

**Response:**
* **GQA Classification:** In GQA models (Llama-3.2, Qwen-2.5), multiple Query heads share KV heads. However, the pre-softmax attention scores ($Q K^T$) are computed independently for *each Query head*. We extract and measure the entropy collapse on the individual output distributions of every Query head (e.g., all 512 in Llama) independently.
* **Internal Geometry:** To visualize the manifold of a head, we extract the Key tensors from `past_key_values` after a single forward pass and apply PCA. For GQA, we project the specific `kv_group` mapped from the Query head index. This visualization shows how specific heads cluster semantically identical tokens in vector space.
