"""
02_phase0_diagnostics.py
HeadGenome 2 -- Phase 0 Post-Hoc Diagnostics

Runs all 6 load-bearing checks on the gate_results.json + feature_bank.csv:

  Check 1: Per-class recall in each LOAO fold (is the gap carried by Local/Sink only?)
  Check 2: Feature ablation -- LOAO gap with each feature dropped (is one feature dominating?)
  Check 3: Normalization leakage -- is StandardScaler fit on held-out data?
           (We re-run LOAO with scaler fit strictly inside the training fold.)
  Check 4: GQA feature ablation -- does removing is_gqa/gqa_* collapse the gap on Qwen/Llama?
  Check 5: Majority-class baseline alongside Model A and B
  Check 6: CI width audit -- flag architectures with wide CIs and explain why

Output: outputs/phase0/diagnostics_report.json  (machine-readable)
        and a human-readable summary printed to stdout
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT               = Path(__file__).resolve().parents[2]
FEATURE_BANK_PATH  = ROOT / "outputs" / "phase0" / "feature_bank.csv"
GATE_JSON_PATH     = ROOT / "outputs" / "phase0" / "gate_results.json"
OUTPUT_PATH        = ROOT / "outputs" / "phase0" / "diagnostics_report.json"

# ---------------------------------------------------------------------------
# Config (must match 01_phase0_gate.py)
# ---------------------------------------------------------------------------
ARCHITECTURES = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]
RANDOM_SEED   = 42
L2_C          = 0.1
N_BOOTSTRAP   = 1000

# Feature columns (must match gate script)
DEPTH_FEATURES = ["rel_depth"]
DEPTH_BINNED_COLS = [f"depth_bin_{i}" for i in range(10)]

BEHAVIORAL_FEATURES = [
    "match_entropy", "delta_collapse", "mean_distance",
    "bos_mass", "local_mass", "long_range_mass",
    "mean_max_attn", "mean_entropy_ss",
    "vq_ratio_atlas", "mean_output_norm",
]

WEIGHT_FEATURES = [
    "wq_norm", "wk_norm", "wv_norm", "wo_norm",
    "vq_ratio_weight",
    "ov_norm", "ov_eff_rank", "ov_sv1", "ov_sv2", "ov_sv3",
    "qk_eff_rank", "qk_sv1", "qk_sv2", "qk_sv3",
]

GQA_FEATURES = ["is_gqa", "gqa_group_size", "gqa_within_group_rank"]

ALL_MODEL_B_FEATURES = (
    DEPTH_FEATURES + BEHAVIORAL_FEATURES + WEIGHT_FEATURES + GQA_FEATURES
)

# Feature groups for ablation
ABLATION_GROUPS = {
    "vq_ratio_weight":  ["vq_ratio_weight"],
    "vq_ratio_atlas":   ["vq_ratio_atlas"],
    "qk_eff_rank":      ["qk_eff_rank", "qk_sv1", "qk_sv2", "qk_sv3"],
    "ov_eff_rank":      ["ov_eff_rank", "ov_sv1", "ov_sv2", "ov_sv3"],
    "GQA_features":     GQA_FEATURES,
    "behavioral_block": BEHAVIORAL_FEATURES,
    "weight_block":     WEIGHT_FEATURES,
    "delta_collapse":   ["delta_collapse"],
    "match_entropy":    ["match_entropy"],
    "mean_distance":    ["mean_distance"],
    "ov_norm":          ["ov_norm"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_binned_depth(df):
    bins = np.linspace(0, 1, 11)
    labels = [f"depth_bin_{i}" for i in range(10)]
    df["depth_bin"] = pd.cut(df["rel_depth"], bins=bins, labels=labels, include_lowest=True)
    dummies = pd.get_dummies(df["depth_bin"], prefix="depth_bin")
    for c in labels:
        if c not in dummies.columns:
            dummies[c] = 0
    return pd.concat([df.reset_index(drop=True), dummies[labels].reset_index(drop=True)], axis=1)


def macro_f1(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f1_score(y_true, y_pred, average="macro", zero_division=0)


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=L2_C, max_iter=1000, multi_class="multinomial",
            solver="lbfgs", random_state=RANDOM_SEED
        )),
    ])


def loao_run(df, y, features, le):
    """
    Run LOAO with a LEAK-FREE scaler: StandardScaler is fit ONLY on the
    training fold and then applied to test. Returns per-arch dict.
    """
    results = {}
    for held_out in ARCHITECTURES:
        train_mask = (df["model"] != held_out).values
        test_mask  = (df["model"] == held_out).values

        X_all = df[features].values.astype(float)
        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_test) == 0:
            continue

        # Scaler fit strictly on training fold only -- no leakage
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = LogisticRegression(
            C=L2_C, max_iter=1000, multi_class="multinomial",
            solver="lbfgs", random_state=RANDOM_SEED
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        mf1 = macro_f1(y_test, y_pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_report(
                le.inverse_transform(y_test),
                le.inverse_transform(y_pred),
                output_dict=True, zero_division=0
            )
        recalls = {cls: report.get(cls, {}).get("recall", 0.0)
                   for cls in ["local", "sink", "induction", "retrieval"]}

        results[held_out] = {
            "macro_f1":         float(mf1),
            "per_class_recall": recalls,
            "n_test":           int(len(y_test)),
            "n_retrieval":      int((df[test_mask]["label"] == "retrieval").sum()),
            "n_induction":      int((df[test_mask]["label"] == "induction").sum()),
            "_y_test":  y_test,
            "_y_pred":  y_pred,
        }
    return results


def loao_gap(res_a, res_b):
    gaps = []
    for arch in ARCHITECTURES:
        if arch in res_a and arch in res_b:
            gaps.append(res_b[arch]["macro_f1"] - res_a[arch]["macro_f1"])
    return gaps


def bootstrap_loao_gap(res_a, res_b, n=N_BOOTSTRAP):
    rng = np.random.RandomState(RANDOM_SEED + 99)
    pooled_yt  = np.concatenate([res_a[a]["_y_test"] for a in ARCHITECTURES if a in res_a])
    pooled_pa  = np.concatenate([res_a[a]["_y_pred"] for a in ARCHITECTURES if a in res_a])
    pooled_pb  = np.concatenate([res_b[a]["_y_pred"] for a in ARCHITECTURES if a in res_b])
    n_total = len(pooled_yt)
    gaps = []
    for _ in range(n):
        idx = rng.choice(n_total, n_total, replace=True)
        try:
            gaps.append(macro_f1(pooled_yt[idx], pooled_pb[idx]) -
                        macro_f1(pooled_yt[idx], pooled_pa[idx]))
        except Exception:
            continue
    g = np.array(gaps)
    return float(np.mean(g)), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("Loading data...")
    df = pd.read_csv(FEATURE_BANK_PATH)
    df = add_binned_depth(df)
    df["label"] = df["label"].str.lower()

    le = LabelEncoder()
    le.fit(["induction", "local", "retrieval", "sink"])
    y = le.transform(df["label"].values)

    # Model A features (depth only, binned)
    features_a = DEPTH_FEATURES + DEPTH_BINNED_COLS
    # Model B features (full)
    features_b = DEPTH_FEATURES + DEPTH_BINNED_COLS + BEHAVIORAL_FEATURES + WEIGHT_FEATURES + GQA_FEATURES

    # =========================================================================
    # Check 1 + 3: Per-class recall in each LOAO fold (leak-free scaler)
    # =========================================================================
    print("\n" + "="*70)
    print("CHECK 1+3: Per-class recall, LOAO (leak-free scaler fitted per fold)")
    print("="*70)
    res_a = loao_run(df, y, features_a, le)
    res_b = loao_run(df, y, features_b, le)

    per_class_table = {}
    for arch in ARCHITECTURES:
        if arch not in res_a:
            continue
        a, b = res_a[arch], res_b[arch]
        per_class_table[arch] = {
            "macro_f1_A": a["macro_f1"],
            "macro_f1_B": b["macro_f1"],
            "gap":        b["macro_f1"] - a["macro_f1"],
            "n_test":     a["n_test"],
            "n_retrieval": a["n_retrieval"],
            "n_induction": a["n_induction"],
            "per_class_A": a["per_class_recall"],
            "per_class_B": b["per_class_recall"],
        }
        print(f"\n  {arch}  (n={a['n_test']}, n_induction={a['n_induction']}, n_retrieval={a['n_retrieval']})")
        print(f"  {'Class':12s}  {'Model A':>8}  {'Model B':>8}  {'delta':>8}")
        for cls in ["local", "sink", "induction", "retrieval"]:
            ra = a["per_class_recall"].get(cls, float("nan"))
            rb = b["per_class_recall"].get(cls, float("nan"))
            delta = rb - ra
            flag = "  <<< NEAR ZERO" if rb < 0.1 else ""
            print(f"  {cls:12s}  {ra:>8.3f}  {rb:>8.3f}  {delta:>+8.3f}{flag}")

    gaps = loao_gap(res_a, res_b)
    print(f"\n  LOAO gaps (leak-free): {[f'{g:+.3f}' for g in gaps]}")
    print(f"  Mean: {np.mean(gaps):+.3f}, Min: {min(gaps):+.3f}, Max: {max(gaps):+.3f}")
    g_mean, g_lo, g_hi = bootstrap_loao_gap(res_a, res_b)
    print(f"  Pooled bootstrap CI: {g_mean:+.3f}  95% CI [{g_lo:+.3f}, {g_hi:+.3f}]  excl. zero: {g_lo > 0}")

    # =========================================================================
    # Check 4: GQA feature ablation
    # =========================================================================
    print("\n" + "="*70)
    print("CHECK 4: GQA feature ablation")
    print("="*70)
    features_no_gqa = [f for f in features_b if f not in GQA_FEATURES]
    res_b_nogqa = loao_run(df, y, features_no_gqa, le)
    gaps_nogqa = loao_gap(res_a, res_b_nogqa)
    for arch, g, g_nogqa in zip(ARCHITECTURES, gaps, gaps_nogqa):
        print(f"  {arch:15s}: gap with GQA={g:+.3f}  gap without GQA={g_nogqa:+.3f}  "
              f"delta={g_nogqa-g:+.3f}")
    gqa_ablation = {
        arch: {"gap_with_gqa": g, "gap_without_gqa": g2, "delta": g2 - g}
        for arch, g, g2 in zip(ARCHITECTURES, gaps, gaps_nogqa)
    }

    # =========================================================================
    # Check 2: Feature ablation (drop-one-group)
    # =========================================================================
    print("\n" + "="*70)
    print("CHECK 2: Feature ablation (drop-one-group, LOAO)")
    print("="*70)
    ablation_results = {}
    baseline_gaps = gaps  # from leak-free run above

    for group_name, group_cols in ABLATION_GROUPS.items():
        feat_ablated = [f for f in features_b if f not in group_cols]
        # skip if ablation removes all features
        if len(feat_ablated) == 0:
            continue
        res_abl = loao_run(df, y, feat_ablated, le)
        abl_gaps = loao_gap(res_a, res_abl)
        mean_drop = float(np.mean(baseline_gaps)) - float(np.mean(abl_gaps))
        ablation_results[group_name] = {
            "gaps_after_ablation": abl_gaps,
            "mean_gap_after":  float(np.mean(abl_gaps)),
            "mean_gap_before": float(np.mean(baseline_gaps)),
            "mean_gap_drop":   mean_drop,
            "pct_drop":        100.0 * mean_drop / (float(np.mean(baseline_gaps)) + 1e-12),
        }
        print(f"  Drop {group_name:22s}: mean gap {np.mean(abl_gaps):+.3f}  "
              f"(was {np.mean(baseline_gaps):+.3f})  "
              f"drop={mean_drop:+.3f}  ({100.*mean_drop/(np.mean(baseline_gaps)+1e-12):.1f}%)")

    # =========================================================================
    # Check 5: Majority-class baseline
    # =========================================================================
    print("\n" + "="*70)
    print("CHECK 5: Majority-class baseline vs Model A vs Model B (LOAO, macro-F1)")
    print("="*70)
    baseline_stats = {}
    for arch in ARCHITECTURES:
        if arch not in res_a:
            continue
        train_mask = (df["model"] != arch).values
        test_mask  = (df["model"] == arch).values
        y_train = y[train_mask]
        y_test  = y[test_mask]

        dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dummy.fit(np.zeros((len(y_train), 1)), y_train)
        y_dummy = dummy.predict(np.zeros((len(y_test), 1)))
        dummy_f1 = macro_f1(y_test, y_dummy)
        dummy_acc = float(np.mean(y_test == y_dummy))

        baseline_stats[arch] = {
            "dummy_macro_f1":   float(dummy_f1),
            "dummy_accuracy":   dummy_acc,
            "model_a_macro_f1": res_a[arch]["macro_f1"],
            "model_b_macro_f1": res_b[arch]["macro_f1"],
        }
        print(f"  {arch:15s}  Dummy={dummy_f1:.3f} (acc={dummy_acc:.3f})  "
              f"A={res_a[arch]['macro_f1']:.3f}  B={res_b[arch]['macro_f1']:.3f}")

    # =========================================================================
    # Check 6: CI width audit
    # =========================================================================
    print("\n" + "="*70)
    print("CHECK 6: Per-architecture bootstrap CI width audit")
    print("="*70)
    rng = np.random.RandomState(RANDOM_SEED + 200)
    ci_audit = {}
    for arch in ARCHITECTURES:
        if arch not in res_a or arch not in res_b:
            continue
        yt  = res_a[arch]["_y_test"]
        pa  = res_a[arch]["_y_pred"]
        pb  = res_b[arch]["_y_pred"]
        n   = len(yt)
        g_boots = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, n, replace=True)
            try:
                g_boots.append(macro_f1(yt[idx], pb[idx]) - macro_f1(yt[idx], pa[idx]))
            except Exception:
                continue
        g_boots = np.array(g_boots)
        lo, hi = float(np.percentile(g_boots, 2.5)), float(np.percentile(g_boots, 97.5))
        width = hi - lo
        n_ret = res_a[arch]["n_retrieval"]
        n_ind = res_a[arch]["n_induction"]
        ci_audit[arch] = {
            "mean_gap": float(np.mean(g_boots)),
            "ci_lower": lo, "ci_upper": hi,
            "ci_width": width,
            "n_test": n,
            "n_retrieval": n_ret,
            "n_induction": n_ind,
            "excludes_zero": lo > 0,
        }
        print(f"  {arch:15s}: [{lo:+.3f}, {hi:+.3f}]  width={width:.3f}  "
              f"n_ret={n_ret}  n_ind={n_ind}  excl0={lo>0}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Key questions answered")
    print("="*70)

    # Q1: Is gap driven by easy classes only?
    any_b_near_zero = False
    for arch, row in per_class_table.items():
        for cls in ["induction", "retrieval"]:
            if row["per_class_B"].get(cls, 0) < 0.10:
                any_b_near_zero = True
                print(f"  [!] {arch}: Model B recall for {cls} = {row['per_class_B'].get(cls,0):.3f} (near zero)")

    if not any_b_near_zero:
        print("  [OK] No LOAO fold has Model B recall near zero for Induction or Retrieval.")

    # Q2: Is one feature dominating?
    if ablation_results:
        dominant = max(ablation_results.items(), key=lambda x: x[1]["pct_drop"])
        print(f"  [!] Dominant feature group: '{dominant[0]}' — "
              f"dropping it reduces mean LOAO gap by {dominant[1]['pct_drop']:.1f}%")
        if dominant[1]["pct_drop"] > 50:
            print(f"       WARNING: >50% of the gap comes from one group. "
                  f"The depth-deconfound claim needs scrutiny.")

    # Q4: GQA leak?
    for arch in ["Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]:
        if arch in gqa_ablation:
            d = gqa_ablation[arch]["delta"]
            if d < -0.05:
                print(f"  [!] {arch}: removing GQA features drops gap by {d:+.3f} — possible arch leakage.")
            else:
                print(f"  [OK] {arch}: GQA ablation delta = {d:+.3f} — GQA features not leaking.")

    # =========================================================================
    # Save JSON
    # =========================================================================
    def sanitise(d):
        if isinstance(d, dict):
            return {k: sanitise(v) for k, v in d.items() if not k.startswith("_")}
        if isinstance(d, (np.integer,)): return int(d)
        if isinstance(d, (np.floating,)): return float(d)
        if isinstance(d, (np.ndarray,)): return d.tolist()
        if isinstance(d, list): return [sanitise(x) for x in d]
        return d

    output = {
        "check1_per_class_recall": sanitise(per_class_table),
        "check2_feature_ablation": ablation_results,
        "check3_normalization_leakage": {
            "note": "LOAO scaler now fit strictly within training fold. "
                    "Gaps recomputed with leak-free scaler.",
            "loao_gaps_leak_free": gaps,
            "mean_gap":   float(np.mean(gaps)),
            "bootstrap_pooled_gap": g_mean,
            "bootstrap_ci_95": [g_lo, g_hi],
            "ci_excludes_zero": g_lo > 0,
        },
        "check4_gqa_ablation": gqa_ablation,
        "check5_majority_baseline": baseline_stats,
        "check6_ci_width_audit": ci_audit,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Diagnostics saved to {OUTPUT_PATH}]")


if __name__ == "__main__":
    run()
