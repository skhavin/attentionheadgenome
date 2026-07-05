import json
import numpy as np
import scipy.stats as stats
import os

MODELS = ["gpt2-medium", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Llama-3.2-1B"]
print("=== RIGOROUS PHASE 2 DATA ANALYSIS ===")

# --- Compute Base-Rate of Punctuation (Law 16) ---
# UD-EWT dataset was used in step3. We can just load it and count punctuation tokens to get a base rate.
with open("outputs/phase2_atlas/dataset.json") as f:
    dataset = json.load(f)

total_tokens = 0
comma_tokens = 0
period_tokens = 0
for sentence in dataset["ud_ewt"]:
    for token in sentence["tokens"]:
        total_tokens += 1
        if token == ",":
            comma_tokens += 1
        elif token == ".":
            period_tokens += 1

comma_base_rate = comma_tokens / total_tokens if total_tokens else 0
period_base_rate = period_tokens / total_tokens if total_tokens else 0
punct_base_rate = comma_base_rate + period_base_rate

print(f"\n[Dataset Base Rates] Total tokens: {total_tokens}, Commas: {comma_base_rate*100:.2f}%, Periods: {period_base_rate*100:.2f}%")

def partial_correlation(x, y, z):
    # Partial correlation of x and y controlling for z
    r_xy = stats.pearsonr(x, y)[0]
    r_xz = stats.pearsonr(x, z)[0]
    r_yz = stats.pearsonr(y, z)[0]
    num = r_xy - r_xz * r_yz
    den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if den == 0: return 0
    return num / den

def partial_corr_p(r, n, k=1):
    if abs(r) == 1.0: return 0.0
    df = n - 2 - k
    t = r * np.sqrt(df / (1 - r**2))
    p = 2 * stats.t.sf(np.abs(t), df)
    return p

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0: return 0.0
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pool_sd == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pool_sd

for m in MODELS:
    path = f"outputs/phase2_atlas/{m}_head_atlas.json"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    
    heads = list(data["heads"].values())
    print(f"\n{'='*60}")
    print(f"MODEL: {m}")
    print(f"{'='*60}")

    # -------------------------------------------------------------
    # LAW 1: V/Q Scaling & Output Norm (Correlation + Permutation)
    # -------------------------------------------------------------
    vq_ratios = []
    out_norms = []
    layers = []
    for h in heads:
        vq = h.get("vq_ratio")
        out = h.get("mean_output_norm")
        lay = h.get("layer")
        if vq is not None and out is not None and lay is not None:
            vq_ratios.append(vq)
            out_norms.append(out)
            layers.append(lay)
            
    if len(vq_ratios) > 10:
        r, p = stats.pearsonr(vq_ratios, out_norms)
        # Permutation Null
        n_permutations = 10000
        null_dist = []
        out_norms_arr = np.array(out_norms)
        for _ in range(n_permutations):
            shuffled = np.random.permutation(out_norms_arr)
            null_r = stats.pearsonr(vq_ratios, shuffled)[0]
            null_dist.append(null_r)
        
        perm_p = np.sum(np.abs(null_dist) >= np.abs(r)) / n_permutations
        partial_r = partial_correlation(vq_ratios, out_norms, layers)
        part_p = partial_corr_p(partial_r, len(vq_ratios), k=1)
        
        print("\n--- Law 1: V/Q Scaling (Statistical Proof) ---")
        print(f"Pearson r(V/Q, OutputNorm): {r:.4f}")
        print(f"Permutation p-value: {perm_p:.5f}")
        print(f"Partial Correlation (controlling for Layer): {partial_r:.4f} (p-value: {part_p:.5f})")

    # -------------------------------------------------------------
    # LAW 11: Softmax Saturation (T-Test & Cohen's d)
    # -------------------------------------------------------------
    retrieval_sat = []
    local_sat = []
    for h in heads:
        # Fixed bug: correctly referencing mean_max_attn
        if "softmax_saturation" in h and "mean_max_attn" in h["softmax_saturation"]:
            val = h["softmax_saturation"]["mean_max_attn"]
            if h["class_label"] == "Retrieval":
                retrieval_sat.append(val)
            elif h["class_label"] == "Local":
                local_sat.append(val)
                
    print("\n--- Law 11: Softmax Saturation ---")
    if len(retrieval_sat) > 0 and len(local_sat) > 0:
        t_stat, p_val = stats.ttest_ind(retrieval_sat, local_sat, equal_var=False)
        d = cohens_d(retrieval_sat, local_sat)
        print(f"Retrieval (N={len(retrieval_sat)}) mean max_attn: {np.mean(retrieval_sat):.4f}")
        print(f"Local     (N={len(local_sat)}) mean max_attn: {np.mean(local_sat):.4f}")
        print(f"T-test p-value: {p_val:.5f}, Cohen's d: {d:.2f}")
    else:
        print(f"Not enough data for T-test. Retrieval N={len(retrieval_sat)}, Local N={len(local_sat)}")

    # -------------------------------------------------------------
    # LAW 16: Punctuation Mini-Sinks (Z-test against Base Rate)
    # -------------------------------------------------------------
    print("\n--- Law 16: Punctuation Z-Test ---")
    # Finding heads that allocate mass to punctuation
    # Our grammar map just maps "punct", we don't have it broken down by comma vs period unfortunately.
    # We will test against the combined punct_base_rate.
    max_punct_mass = 0
    max_punct_head = None
    for h in heads:
        if "grammar_profile" in h and "punct" in h["grammar_profile"]:
            val = h["grammar_profile"]["punct"]
            if val > max_punct_mass:
                max_punct_mass = val
                max_punct_head = h
                
    if max_punct_head:
        # The head assigns 'max_punct_mass' proportion of attention to punctuation.
        # We model this as a binomial proportion test: is p_head > p_base?
        # N = number of attention decisions (approx = number of tokens evaluated). We evaluated 100 sentences * avg len
        N_samples = total_tokens
        z_stat = (max_punct_mass - punct_base_rate) / np.sqrt((punct_base_rate * (1 - punct_base_rate)) / N_samples)
        p_val_z = stats.norm.sf(z_stat) # one-tailed
        print(f"Top punct head L{max_punct_head['layer']}H{max_punct_head['head']}: {max_punct_mass*100:.1f}% mass")
        print(f"Z-statistic vs Base Rate ({punct_base_rate*100:.1f}%): {z_stat:.2f}, p-value: {p_val_z:.2e}")
