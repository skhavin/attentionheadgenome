# phase1/step8_relabel_gpt2.py
#
# PURPOSE:
#   Re-label GPT-2 heads using mechanistically-grounded thresholds from the
#   entropy-collapse experiment (step7), replacing the histogram-proxy KMeans labels.
#
# LABELING RULES (empirically derived):
#   sink:      NaN entropy OR (match_entropy < 0.1 AND nonmatch_entropy < 0.1)
#   retrieval: delta > +0.30 nats
#   induction: delta < -0.50 nats
#   local:     everything else (near-zero delta, mid-entropy)
#
# OUTPUTS:
#   outputs/phase1/gpt2_mechanistic_labels.json   — per-head role assignments
#   prints spatial depth distribution per role     — Spatial Law check
#   prints KMeans overlap                          — taxonomy validation

import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")

ENTROPY_JSON = os.path.join(IN_DIR, "gpt2_retrieval_entropy.json")
SUMMARY_JSON = os.path.join(IN_DIR, "gpt2-medium_patterns_summary.json")
OUT_JSON     = os.path.join(IN_DIR, "gpt2_mechanistic_labels.json")

# Mechanistic thresholds
THRESHOLD_RETRIEVAL = 0.30   # delta > this -> retrieval
THRESHOLD_INDUCTION = -0.50  # delta < this -> induction
THRESHOLD_SINK_ENT  = 0.10   # entropy < this in both conditions -> sink


def apply_labels(entropy_data):
    """
    Apply mechanistic thresholds to entropy-collapse data.
    Returns dict: "layer_head" -> role
    """
    labels = {}
    counts = {"sink": 0, "retrieval": 0, "induction": 0, "local": 0}

    for key, vals in entropy_data["heads"].items():
        # NaN head (sink — 100% BOS attending, entropy = 0)
        if vals.get("nan") or vals["match_entropy"] is None:
            role = "sink"
        else:
            me    = vals["match_entropy"]
            nme   = vals["nonmatch_entropy"]
            delta = vals["delta"]

            if me < THRESHOLD_SINK_ENT and nme < THRESHOLD_SINK_ENT:
                role = "sink"
            elif delta > THRESHOLD_RETRIEVAL:
                role = "retrieval"
            elif delta < THRESHOLD_INDUCTION:
                role = "induction"
            else:
                role = "local"

        labels[key] = role
        counts[role] += 1

    return labels, counts


def depth_distribution(labels, num_layers=24):
    """
    Bin mechanistically-labeled heads by relative depth.
    Returns dict: role -> list of relative depths
    """
    role_depths = {"sink": [], "retrieval": [], "induction": [], "local": []}
    for key, role in labels.items():
        layer, head = map(int, key.split("_"))
        rel_depth = layer / (num_layers - 1)
        role_depths[role].append(rel_depth)
    return role_depths


def kmeans_overlap(labels, summary_json):
    """
    Run KMeans on the histogram data, compare cluster assignments with
    mechanistic labels to measure how well histogram clusters track roles.
    """
    with open(summary_json) as f:
        data = json.load(f)

    heads = {}
    for key, hist in data["heads"].items():
        layer, head = map(int, key.split("_"))
        heads[(layer, head)] = np.array(hist, dtype=np.float32)

    keys = sorted(heads.keys())
    X    = np.array([heads[k] for k in keys])

    km     = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    sil    = silhouette_score(X, km_labels)

    # For each KMeans cluster, count mechanistic roles inside it
    cluster_role_counts = {}
    for i, (layer, head) in enumerate(keys):
        cluster_id = int(km_labels[i])
        mech_role  = labels.get(str(layer) + "_" + str(head), "unknown")
        if cluster_id not in cluster_role_counts:
            cluster_role_counts[cluster_id] = {"sink": 0, "retrieval": 0, "induction": 0, "local": 0}
        cluster_role_counts[cluster_id][mech_role] += 1

    return cluster_role_counts, sil


def main():
    print("Loading entropy-collapse results...")
    with open(ENTROPY_JSON) as f:
        entropy_data = json.load(f)

    # ── 1. Apply mechanistic labels ──────────────────────────────────────────
    labels, counts = apply_labels(entropy_data)
    total = sum(counts.values())

    print("\n=== GPT-2 Mechanistic Labels (step7 thresholds) ===")
    print("  Thresholds: retrieval > +" + str(THRESHOLD_RETRIEVAL) +
          " nats | induction < " + str(THRESHOLD_INDUCTION) +
          " nats | sink = NaN or ent<" + str(THRESHOLD_SINK_ENT))
    print()
    for role in ["sink", "retrieval", "induction", "local"]:
        n   = counts[role]
        pct = (n / total) * 100
        print("  " + role.ljust(12) + str(n).rjust(4) + "  (" + str(round(pct, 1)) + "%)")

    # ── 2. Spatial Law check ─────────────────────────────────────────────────
    role_depths = depth_distribution(labels)
    print("\n=== Spatial Law: Relative Depth per Role ===")
    print("  (Prediction: sink/local early, retrieval mid-late, induction late)")
    print()
    for role in ["sink", "retrieval", "induction", "local"]:
        depths = role_depths[role]
        if not depths:
            print("  " + role + ": no heads")
            continue
        depths_arr = np.array(depths)
        print("  " + role.ljust(12) +
              "  mean=" + str(round(float(depths_arr.mean()), 3)) +
              "  std=" + str(round(float(depths_arr.std()), 3)) +
              "  min=" + str(round(float(depths_arr.min()), 3)) +
              "  max=" + str(round(float(depths_arr.max()), 3)) +
              "  n=" + str(len(depths)))

    # ── 3. Per-role head list (sorted by depth for retrieval) ────────────────
    print("\n=== Retrieval Heads (sorted by delta desc) ===")
    retrieval_with_delta = []
    for key, role in labels.items():
        if role == "retrieval":
            layer, head = map(int, key.split("_"))
            delta = entropy_data["heads"][key]["delta"]
            me    = entropy_data["heads"][key]["match_entropy"]
            nme   = entropy_data["heads"][key]["nonmatch_entropy"]
            rel_d = layer / 23.0
            retrieval_with_delta.append((delta, layer, head, rel_d, me, nme))
    retrieval_with_delta.sort(reverse=True)
    print("  delta     layer  head  rel_depth  match_ent  nonmatch_ent")
    for delta, layer, head, rel_d, me, nme in retrieval_with_delta:
        print("  " + str(round(delta, 4)).ljust(9) +
              str(layer).ljust(7) + str(head).ljust(6) +
              str(round(rel_d, 3)).ljust(11) +
              str(round(me, 4)).ljust(11) + str(round(nme, 4)))

    print("\n=== Induction Heads (sorted by delta asc) ===")
    induction_with_delta = []
    for key, role in labels.items():
        if role == "induction":
            layer, head = map(int, key.split("_"))
            delta = entropy_data["heads"][key]["delta"]
            rel_d = layer / 23.0
            induction_with_delta.append((delta, layer, head, rel_d))
    induction_with_delta.sort()
    print("  delta     layer  head  rel_depth")
    for delta, layer, head, rel_d in induction_with_delta:
        print("  " + str(round(delta, 4)).ljust(9) +
              str(layer).ljust(7) + str(head).ljust(6) +
              str(round(rel_d, 3)))

    # ── 4. KMeans overlap (cross-validation of the taxonomy) ────────────────
    print("\n=== KMeans Cluster vs Mechanistic Label Overlap ===")
    cluster_role_counts, sil = kmeans_overlap(labels, SUMMARY_JSON)
    print("  KMeans silhouette: " + str(round(sil, 4)))
    print("  (Each row = one KMeans cluster; columns = mechanistic role counts)")
    print()
    print("  Cluster   sink  local  retrieval  induction")
    for cid in sorted(cluster_role_counts.keys()):
        rc = cluster_role_counts[cid]
        row_total = sum(rc.values())
        dominant  = max(rc, key=rc.get)
        print("  C" + str(cid) + "  (n=" + str(row_total) + ") " +
              str(rc["sink"]).rjust(5) +
              str(rc["local"]).rjust(7) +
              str(rc["retrieval"]).rjust(11) +
              str(rc["induction"]).rjust(11) +
              "   -> dominant: " + dominant)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out = {
        "model": "gpt2-medium",
        "thresholds": {
            "retrieval_delta_min": THRESHOLD_RETRIEVAL,
            "induction_delta_max": THRESHOLD_INDUCTION,
            "sink_entropy_max":    THRESHOLD_SINK_ENT,
        },
        "counts": counts,
        "heads": labels,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> " + OUT_JSON)
    print("[DONE]")


if __name__ == "__main__":
    main()
