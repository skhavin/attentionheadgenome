"""
step21_classification_stability.py
------------------------------------
Loads geometry profiles from all 4 domains (output of step20) and
re-classifies every head using the exact same thresholds from step6.

Compares domain-specific labels to the original WikiText atlas.
Produces:
  1. A stability matrix (confusion matrix per class, cross-domain)
  2. Per-head stability scores (fraction of domains that agree with WikiText)
  3. A list of "ambiguous heads" that should be kept at full softmax

Output: outputs/routing/Qwen2.5-0.5B_stability.json
"""
import json, os, numpy as np
from collections import defaultdict

MODEL = "Qwen2.5-0.5B"
DOMAINS = ["wikipedia", "code", "dialogue", "math"]

# Same thresholds as step6_compile_atlas.py
BOS_SINK_THRESH    = 0.50
# Note: Retrieval/Induction are from Phase 1 delta_collapse — those are
# weight-derived, so stable. We only re-test Local vs Sink here.
# A head is re-classified as Sink if bos_mass > threshold on this domain.

print("Loading original WikiText atlas...")
atlas_path = f"outputs/phase2_atlas/{MODEL}_head_atlas.json"
with open(atlas_path) as f:
    atlas = json.load(f)
original_labels = {k: v["class_label"] for k, v in atlas["heads"].items()}

print(f"Total heads in atlas: {len(original_labels)}")

# Load domain geometry files
domain_geometries = {}
for domain in DOMAINS:
    path = f"outputs/routing/{MODEL}_{domain}_geometry.json"
    if os.path.exists(path):
        with open(path) as f:
            domain_geometries[domain] = json.load(f)["heads"]
        print(f"Loaded {domain}: {len(domain_geometries[domain])} heads")
    else:
        print(f"WARNING: Missing {path} — skipping.")

def reclassify(head_key, geometry, original_label):
    """
    Re-classify based on geometry only (bos_mass).
    Retrieval/Induction are determined by entropy delta (Phase 1),
    which is weight-derived and domain-independent — so we keep those.
    We only test Local vs Sink stability.
    """
    if original_label in ("Retrieval", "Induction"):
        return original_label  # weight-derived, stable by construction
    
    g = geometry.get(head_key, {})
    bos = g.get("bos_mass", 0.0)
    if bos > BOS_SINK_THRESH:
        return "Sink"
    return "Local"

# Per-head stability: count how many domains agree with original label
head_stability = {}
domain_labels_per_head = defaultdict(dict)

for head_key, orig_label in original_labels.items():
    agreements = 0
    for domain, geom in domain_geometries.items():
        new_label = reclassify(head_key, geom, orig_label)
        domain_labels_per_head[head_key][domain] = new_label
        if new_label == orig_label:
            agreements += 1
    stability = agreements / len(domain_geometries) if domain_geometries else 0
    head_stability[head_key] = round(stability, 4)

# Stability matrix: for each original class, how often does each domain agree?
print("\n=== STABILITY MATRIX ===")
print(f"{'Class':<12} {'N':>4} {'Wiki':>6} {'Code':>6} {'Dialog':>7} {'Math':>6} {'Mean':>6}")
print("-" * 55)

for cls in ["Local", "Sink", "Induction", "Retrieval"]:
    cls_heads = [k for k, v in original_labels.items() if v == cls]
    n = len(cls_heads)
    if n == 0:
        continue
    
    per_domain_agree = {}
    for domain in DOMAINS:
        if domain not in domain_geometries:
            per_domain_agree[domain] = None
            continue
        agree = sum(1 for k in cls_heads
                    if domain_labels_per_head[k].get(domain) == cls)
        per_domain_agree[domain] = agree / n
    
    vals = [per_domain_agree.get(d) for d in DOMAINS]
    mean_agree = np.mean([v for v in vals if v is not None])
    
    row = f"{cls:<12} {n:>4}"
    for v in vals:
        row += f" {v*100:>5.1f}%" if v is not None else f" {'N/A':>6}"
    row += f" {mean_agree*100:>5.1f}%"
    print(row)

# Ambiguous heads (stability < 85%)
STABILITY_THRESH = 0.85
ambiguous = {k: v for k, v in head_stability.items() if v < STABILITY_THRESH}
stable    = {k: v for k, v in head_stability.items() if v >= STABILITY_THRESH}

print(f"\n=== ROUTING DECISIONS ===")
print(f"Stable heads (>={STABILITY_THRESH*100:.0f}% domain agreement): {len(stable)}")
print(f"Ambiguous heads (kept at FULL_SOFTMAX):                    {len(ambiguous)}")

# Build routing map
routing_map = {}
routing_rules = {
    "Induction": "FULL_SOFTMAX",
    "Retrieval":  "FULL_SOFTMAX",
    "Local":      "WINDOW_32",
    "Sink":       "BOS_ROUTE",
}
for head_key, orig_label in original_labels.items():
    stab = head_stability.get(head_key, 0)
    if stab >= STABILITY_THRESH:
        routing = routing_rules[orig_label]
    else:
        routing = "FULL_SOFTMAX"  # conservative for ambiguous heads
    
    routing_map[head_key] = {
        "class": orig_label,
        "stability": head_stability.get(head_key, 0),
        "routing": routing,
    }

# Summary by routing type
from collections import Counter
route_counts = Counter(v["routing"] for v in routing_map.values())
print("\nRouting budget:")
for rtype, count in sorted(route_counts.items()):
    print(f"  {rtype}: {count} heads ({count/len(routing_map)*100:.1f}%)")

out_path = f"outputs/routing/{MODEL}_stability.json"
os.makedirs("outputs/routing", exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "model": MODEL,
        "stability_threshold": STABILITY_THRESH,
        "routing_map": routing_map,
        "n_stable": len(stable),
        "n_ambiguous": len(ambiguous),
    }, f, indent=2)
print(f"\nSaved routing map to {out_path}")
