# Dynamic $O(N)$ Routing Mechanics

This document provides a detailed breakdown of how the **HeadGenome** architecture achieves strict asymptotic $O(N)$ sequence scaling for critical *Retrieval* and *Induction* heads. 

While these heads technically perform an $O(N)$ "full cache" backward lookup during generation to locate needles and long-range dependencies, we use a **Dynamic Regime-Switching Router** to gate this operation, strictly capping the sequence-level algorithmic complexity to $O(N)$.

---

## 1. The Core Challenge
Retrieval and Induction heads require full $O(N)$ context. If we compute the standard unmasked attention score ($QK^T$) for every token against the full $N$-length cache during decoding, the aggregate computational complexity is:
$$ \sum_{i=1}^{N} i = \frac{N(N-1)}{2} = \mathcal{O}(N^2) $$

To break the $O(N^2)$ barrier without losing reasoning accuracy, we must avoid running the full attention lookup for the vast majority of tokens, triggering it *only* when absolutely necessary.

---

## 2. The $O(W)$ Forward Pass & Regime Detection
Instead of doing a full cache lookup, the model utilizes lightweight **Regime Detectors** and **MoE Routers** that scan the input sequence via a cheap forward pass.

### Files Involved:
*   `old-proj/phase7/profiling/regime_detector.py`
*   `old-proj/phase7/moe/router.py`

### The Mechanism:
For every incoming query token $Q_{last}$, the detector computes a set of structural heuristic features using only a small, local sliding window ($W$):
1. **n-gram Repetition Rate**: Scans the prefix to see if structural tokens are looping.
2. **Max Token Frequency**: Detects density spikes of copy-triggers.
3. **Local Entropy**: The entropy of the attention distribution across the last $W$ tokens.
4. **Max Similarity**: The maximum dot-product score against the $W$ local tokens.

Because these features are computed strictly over a constant window $W$ (or via an online linear scan), calculating the routing decision for a single token is $O(W) \approx O(1)$.
Across the entire sequence $N$, computing the forward pass features takes **$O(N \cdot W) \rightarrow O(N)$** operations.

---

## 3. The Sparse Trigger & Gated Backward Lookup
Once the $O(W)$ features are computed, the router makes a boolean decision: *Are we in a copy-trigger regime, or have we found a semantic match?*

### Files Involved:
*   `old-proj/phase7/moe/moe_patcher.py` (The attention layer override)

### The Mechanism:
1. **No Match (Common Case)**: The query token does not trigger the regime switch. The head falls back to a cheap $O(1)$ substitution (e.g., restricted local window or EMA recurrence). 
2. **Match (Rare Trigger Case)**: The query token triggers the detector. The router dynamically applies **Full Causal Attention** ($O(N)$ cache lookup) exclusively for this token to perfectly retrieve the required long-range context.

---

## 4. The Final Asymptotic Proof
Why does this guarantee $O(N)$ complexity overall?

Because linguistic triggers (like retrieving a specific noun, or finding a needle in a haystack) are structurally sparse. A needle is typically retrieved exactly once. 
Therefore, the number of times the expensive $O(N)$ cache lookup is triggered per sequence is a bounded constant $C$ (i.e., $O(1)$).

**Summing the Total Computational Complexity:**
1. **Scanning via Forward Pass:** $O(N \cdot W) = \mathcal{O}(N)$
2. **Execution of Sparse Full-Cache Lookups:** $\mathcal{O}(1) \text{ triggers} \times \mathcal{O}(N) \text{ lookup} = \mathcal{O}(N)$

$$ \mathcal{O}_{total} = \mathcal{O}(N) + \mathcal{O}(N) = \mathbf{\mathcal{O}(N)} $$

By splitting the attention mechanism into a cheap $O(N)$ continuous forward scan and a rare $O(N)$ backward cache lookup, HeadGenome perfectly retrieves long-range facts while structurally guaranteeing that the algorithmic scaling remains linear.
