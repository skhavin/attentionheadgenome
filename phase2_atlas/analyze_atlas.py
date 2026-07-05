import json
import numpy as np

MODELS = ["gpt2-medium", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Llama-3.2-1B"]

print("=== PHASE 2 ATLAS DATA ANALYSIS ===\n")

for m in MODELS:
    try:
        with open(f"outputs/phase2_atlas/{m}_head_atlas.json") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {m}: {e}")
        continue
    
    heads = data["heads"].values()
    print(f"\n{'='*40}")
    print(f"MODEL: {m}")
    print(f"{'='*40}")

    # 1. LAW 1: V/Q Scaling Law (Deeper heads have higher V/Q ratio -> higher output norm)
    # Let's bucket into early (first third), mid (middle third), late (last third)
    L = max(h["layer"] for h in heads) + 1
    early = [h for h in heads if h["layer"] < L // 3]
    mid   = [h for h in heads if L // 3 <= h["layer"] < (2 * L) // 3]
    late  = [h for h in heads if h["layer"] >= (2 * L) // 3]

    def avg(lst, key): 
        vals = [h.get(key, 0) for h in lst if h.get(key) is not None]
        return np.mean(vals) if vals else 0.0

    print("\n--- Law 1: V/Q Scaling & Output Norm ---")
    print(f"Early Layers: V/Q = {avg(early, 'vq_ratio'):.3f}, OutNorm = {avg(early, 'mean_output_norm'):.3f}")
    print(f"Mid Layers:   V/Q = {avg(mid, 'vq_ratio'):.3f}, OutNorm = {avg(mid, 'mean_output_norm'):.3f}")
    print(f"Late Layers:  V/Q = {avg(late, 'vq_ratio'):.3f}, OutNorm = {avg(late, 'mean_output_norm'):.3f}")

    # 2. LAW 11: Softmax Saturation (Retrieval/Induction saturate, Local is diffuse)
    print("\n--- Law 11: Softmax Saturation ---")
    classes = {"Sink": [], "Local": [], "Induction": [], "Retrieval": []}
    for h in heads:
        c = h["class_label"]
        if c in classes:
            classes[c].append(h)
            
    for c, hl in classes.items():
        if not hl: continue
        # Softmax saturation measured by mean of max_attn over prompts
        sat_vals = []
        for h in hl:
            if "softmax_saturation" in h and "max_attn" in h["softmax_saturation"]:
                sat_vals.append(np.mean(h["softmax_saturation"]["max_attn"]))
        avg_sat = np.mean(sat_vals) if sat_vals else 0.0
        print(f"{c:10s} (N={len(hl):3d}): Avg Max Attention = {avg_sat:.3f}")

    # 3. PILLAR 3: Syntax Specialization (Grammar Map)
    print("\n--- Pillar 3: Universal Dependencies Grammar Specialization ---")
    nsubj_masses = []
    obj_masses = []
    punct_masses = []
    for h in heads:
        if "grammar_profile" in h:
            nsubj_masses.append((h["layer"], h["head"], h["grammar_profile"].get("nsubj", 0)))
            obj_masses.append((h["layer"], h["head"], h["grammar_profile"].get("obj", 0)))
            punct_masses.append((h["layer"], h["head"], h["grammar_profile"].get("punct", 0)))
    
    if nsubj_masses:
        nsubj_masses.sort(key=lambda x: x[2], reverse=True)
        obj_masses.sort(key=lambda x: x[2], reverse=True)
        punct_masses.sort(key=lambda x: x[2], reverse=True)
        print(f"Top nsubj (subject) head: L{nsubj_masses[0][0]}H{nsubj_masses[0][1]} ({nsubj_masses[0][2]:.3f} mass)")
        print(f"Top obj (object) head:    L{obj_masses[0][0]}H{obj_masses[0][1]} ({obj_masses[0][2]:.3f} mass)")
        print(f"Top punct (comma) head:   L{punct_masses[0][0]}H{punct_masses[0][1]} ({punct_masses[0][2]:.3f} mass)")

    # 4. PILLAR 4: Causal Sink Falsification
    print("\n--- Pillar 4: Causal Sink Falsification (BOS Removal) ---")
    if classes["Sink"]:
        deltas = []
        for h in classes["Sink"]:
            if "sink_falsification" in h:
                # delta_no_bos is the absolute entropy change when BOS is removed
                deltas.append(h["sink_falsification"].get("delta_no_bos", 0))
        avg_delta = np.mean(deltas) if deltas else 0.0
        max_delta = np.max(deltas) if deltas else 0.0
        print(f"Sink Heads (N={len(classes['Sink'])}) -> Avg Entropy Explosion = {avg_delta:.3f} (Max: {max_delta:.3f})")
