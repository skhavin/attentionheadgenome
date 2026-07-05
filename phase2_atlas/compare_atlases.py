import json
import os
import numpy as np

models = ["gpt2-medium", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Llama-3.2-1B"]
atlas_dir = "outputs/phase2_atlas"

atlases = {}
for m in models:
    path = os.path.join(atlas_dir, f"{m}_head_atlas.json")
    if os.path.exists(path):
        with open(path) as f:
            atlases[m] = json.load(f)

print("=== CROSS-MODEL ATLAS COMPARISON ===")

# 1. Class Distribution
print("\n--- Head Class Distribution ---")
for m, data in atlases.items():
    classes = {}
    total = len(data["heads"])
    for h in data["heads"].values():
        c = h.get("class_label", "Unknown")
        classes[c] = classes.get(c, 0) + 1
    
    print(f"{m} (N={total}):")
    for c, count in sorted(classes.items()):
        print(f"  {c}: {count} ({count/total*100:.1f}%)")

# 2. Normalized Depth of Classes
print("\n--- Normalized Depth by Class ---")
for m, data in atlases.items():
    max_layer = max(h["layer"] for h in data["heads"].values())
    depths = {}
    for h in data["heads"].values():
        c = h.get("class_label", "Unknown")
        layer = h["layer"]
        norm_depth = layer / max_layer if max_layer > 0 else 0
        if c not in depths:
            depths[c] = []
        depths[c].append(norm_depth)
        
    print(f"{m}:")
    for c in sorted(depths.keys()):
        arr = depths[c]
        print(f"  {c}: mean depth = {np.mean(arr):.2f}, std = {np.std(arr):.2f}")

# 3. Structural V/Q Ratio Trends
print("\n--- V/Q Ratio averages by layer depth ---")
for m, data in atlases.items():
    max_layer = max(h["layer"] for h in data["heads"].values())
    early = []
    mid = []
    late = []
    for h in data["heads"].values():
        layer = h["layer"]
        vq = h["vq_ratio"]
        if layer < max_layer / 3:
            early.append(vq)
        elif layer < 2 * max_layer / 3:
            mid.append(vq)
        else:
            late.append(vq)
            
    print(f"{m}: Early={np.mean(early):.2f}, Mid={np.mean(mid):.2f}, Late={np.mean(late):.2f}")

# 4. Syntactic Tracking (Grammar)
print("\n--- Syntactic Specialization (nsubj vs obj) ---")
for m, data in atlases.items():
    max_nsubj = 0
    max_obj = 0
    for h in data["heads"].values():
        if "grammar_profile" in h:
            prof = h["grammar_profile"]
            if "nsubj" in prof and prof["nsubj"] > max_nsubj:
                max_nsubj = prof["nsubj"]
            if "obj" in prof and prof["obj"] > max_obj:
                max_obj = prof["obj"]
    print(f"{m}: Max nsubj mass = {max_nsubj*100:.1f}%, Max obj mass = {max_obj*100:.1f}%")

# 5. Punctuation Sinks
print("\n--- Punctuation Sink (Mini-sink) Max Mass ---")
for m, data in atlases.items():
    max_punct = 0
    for h in data["heads"].values():
        if "grammar_profile" in h and "punct" in h["grammar_profile"]:
            val = h["grammar_profile"]["punct"]
            if val > max_punct:
                max_punct = val
    print(f"{m}: Max punct mass = {max_punct*100:.1f}%")
