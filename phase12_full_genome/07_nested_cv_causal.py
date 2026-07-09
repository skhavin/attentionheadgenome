"""
07_nested_cv_causal.py

Nested Cross-Validation for Head Function Classification based on CAUSAL MECHANISTIC Features.

Features:
- Q-Permutation \u0394PPL (Condition A)
- Zero-OV \u0394PPL (Condition B)
- Position Shuffle \u0394PPL
- Content Shuffle \u0394PPL

Outer Loop: Leave-One-Architecture-Out (LOAO) to test strict zero-shot generalization.
Inner Loop: Stratified K-Fold on the remaining architectures for hyperparameter tuning.

Metric: Macro-F1 score across the target classes.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

RESULTS_03 = "03_qk_ov_ablation_results.json"
RESULTS_04 = "04_shuffle_survival_results.json"
CANONICAL_LABELS = "../outputs/canonical_labels.json"
OUTPUT_JSON = "07_nested_cv_causal_results.json"

def run_nested_cv():
    print("="*60)
    print("GATE B: NESTED CROSS-VALIDATION ON CAUSAL FEATURES")
    print("="*60)
    
    if not os.path.exists(RESULTS_03) or not os.path.exists(RESULTS_04) or not os.path.exists(CANONICAL_LABELS):
        print("Missing required input JSON files.")
        return
        
    with open(RESULTS_03, "r") as f:
        data_03 = json.load(f)
        
    with open(RESULTS_04, "r") as f:
        data_04 = json.load(f)
        
    with open(CANONICAL_LABELS, "r") as f:
        labels_data = json.load(f)
        
    # Extract labels
    labels_dict = {}
    for arch, arch_data in labels_data.get("models", {}).items():
        for head_key, head_info in arch_data.get("heads", {}).items():
            label = head_info.get("label", "unknown")
            if label != "unknown":
                labels_dict[f"{arch}_{head_key}"] = label
                
    # Build feature dataset
    features = []
    
    # Process 03 features (Condition A and B)
    res_A = data_03.get("raw_results", {}).get("condition_A_q_permute", {})
    res_B = data_03.get("raw_results", {}).get("condition_B_zero_ov", {})
    
    res_pos = data_04.get("position_shuffle", {})
    res_con = data_04.get("content_shuffle", {})
    
    archs = set(res_A.keys()) | set(res_B.keys()) | set(res_pos.keys()) | set(res_con.keys())
    
    for arch in archs:
        # Group by head
        head_dict = {}
        
        def add_feature(arch_res, key_name):
            for r in arch_res.get(arch, []):
                head_key = f"{r['layer_idx']}_{r['head_idx']}"
                if head_key not in head_dict:
                    head_dict[head_key] = {"model_name": arch, "layer": r["layer_idx"], "head": r["head_idx"]}
                head_dict[head_key][key_name] = r["delta_ppl"]
                
        add_feature(res_A, "delta_ppl_q_permute")
        add_feature(res_B, "delta_ppl_zero_ov")
        add_feature(res_pos, "delta_ppl_pos_shuffle")
        add_feature(res_con, "delta_ppl_content_shuffle")
        
        for head_key, head_features in head_dict.items():
            global_key = f"{arch}_{head_key}"
            if global_key in labels_dict:
                # Ensure all 4 features are present
                if all(k in head_features for k in ["delta_ppl_q_permute", "delta_ppl_zero_ov", "delta_ppl_pos_shuffle", "delta_ppl_content_shuffle"]):
                    head_features["canonical_label"] = labels_dict[global_key]
                    features.append(head_features)
                    
    df = pd.DataFrame(features)
    
    if len(df) == 0:
        print("Error: No labeled data found with all 4 causal features.")
        return
        
    print(f"Loaded {len(df)} labeled heads with complete causal features.")
    print("Class distribution:")
    print(df["canonical_label"].value_counts())
    
    feature_cols = [
        "delta_ppl_q_permute", 
        "delta_ppl_zero_ov", 
        "delta_ppl_pos_shuffle", 
        "delta_ppl_content_shuffle"
    ]
    
    models = df["model_name"].unique()
    print(f"\nArchitectures for LOAO: {list(models)}\n")
    
    results = {}
    
    for test_model in models:
        print(f"--- Holding out: {test_model} ---")
        
        train_df = df[df["model_name"] != test_model]
        test_df = df[df["model_name"] == test_model]
        
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  Skipping {test_model} due to insufficient split data.")
            continue
            
        X_train = train_df[feature_cols].values
        y_train = train_df["canonical_label"].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df["canonical_label"].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning via GridSearchCV (Logistic Regression)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight': ['balanced']
        }
        
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=2000, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_clf = grid_search.best_estimator_
        
        print(f"  Best params: {grid_search.best_params_}")
        
        # Test on held-out architecture
        y_pred = best_clf.predict(X_test_scaled)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
        
        print(f"  Zero-Shot Macro-F1: {macro_f1:.4f}")
        
        # Display Feature Importances (coefficients)
        print("  Feature Coefficients:")
        coefs = best_clf.coef_
        for idx, class_name in enumerate(best_clf.classes_):
            class_coefs = coefs[0] if len(best_clf.classes_) == 2 else coefs[idx]
            top_idx = np.argsort(np.abs(class_coefs))[::-1]
            print(f"    {class_name}:")
            for i in top_idx[:3]:  # Top 3 features
                print(f"      {feature_cols[i]}: {class_coefs[i]:.4f}")
            if len(best_clf.classes_) == 2: break
            
        results[test_model] = {
            "macro_f1": macro_f1,
            "report": report,
            "confusion_matrix": cm.tolist(),
            "classes": best_clf.classes_.tolist(),
            "best_params": grid_search.best_params_
        }
        print()
        
    # Aggregate summary
    f1_scores = [r["macro_f1"] for r in results.values()]
    mean_f1 = np.mean(f1_scores)
    
    print("="*60)
    print(f"FINAL LOAO MACRO-F1 (CAUSAL): {mean_f1:.4f}")
    print("="*60)
    
    # Add passing criteria for Gate B
    gate_b_pass = mean_f1 >= 0.70
    print(f"\nGATE B VERDICT: {'PASS' if gate_b_pass else 'FAIL'}")
    
    summary = {
        "mean_macro_f1": mean_f1,
        "per_architecture": results,
        "gate_b_pass": gate_b_pass
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_nested_cv()
