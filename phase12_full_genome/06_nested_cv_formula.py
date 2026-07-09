"""
06_nested_cv_formula.py

Nested Cross-Validation for Head Function Classification based on Static Weight Features.

Outer Loop: Leave-One-Architecture-Out (LOAO) to test strict zero-shot generalization.
Inner Loop: Stratified K-Fold on the remaining architectures for hyperparameter tuning.

Metric: Macro-F1 score across the target classes.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json

DATASET_PATH = "full_genome_dataset.csv"
OUTPUT_JSON = "06_nested_cv_results.json"

def run_nested_cv():
    print("="*60)
    print("GATE B: NESTED CROSS-VALIDATION (LOAO)")
    print("="*60)
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset {DATASET_PATH} not found.")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # Filter only known labels
    df = df[df["canonical_label"] != "unknown"].copy()
    
    if len(df) == 0:
        print("Error: No labeled data found.")
        return
        
    print(f"Loaded {len(df)} labeled heads.")
    print("Class distribution:")
    print(df["canonical_label"].value_counts())
    
    # Define feature columns (exclude metadata and labels)
    exclude_cols = [
        "model_name", "model_id", "canonical_label", "layer_index", "head_index",
        # We also exclude highly architecture-specific absolute numbers if needed, 
        # but relative_depth and normalized features are okay.
        "num_layers", "num_attention_heads", "num_kv_heads", "head_dim"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\nUsing {len(feature_cols)} static weight features.")
    
    models = df["model_name"].unique()
    print(f"Architectures for LOAO: {list(models)}\n")
    
    results = {}
    
    # Outer Loop: Leave-One-Architecture-Out
    for test_model in models:
        print(f"--- Holding out: {test_model} ---")
        
        train_df = df[df["model_name"] != test_model]
        test_df = df[df["model_name"] == test_model]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["canonical_label"].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df["canonical_label"].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inner Loop: Hyperparameter tuning via GridSearchCV
        # We use a simple Logistic Regression with L2 penalty to measure linear separability,
        # but Random Forest can also be tested. Let's use Logistic Regression as the baseline formula.
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
        
        print("  Running inner-loop Grid Search...")
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  Best params: {grid_search.best_params_}")
        
        # Evaluate on held-out architecture
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        print(f"  Test Macro-F1: {macro_f1:.4f}")
        
        results[test_model] = {
            "macro_f1": float(macro_f1),
            "classification_report": report,
            "best_params": grid_search.best_params_
        }
        
    print("\n" + "="*60)
    print("GATE B SUMMARY")
    print("="*60)
    
    mean_f1 = np.mean([r["macro_f1"] for r in results.values()])
    print(f"Mean LOAO Macro-F1 across all architectures: {mean_f1:.4f}")
    
    if mean_f1 > 0.65:
        print("\n[PASS] Gate B Cleared: Static weight features generalize across architectures!")
    else:
        print("\n[FAIL] Gate B Failed: Poor zero-shot generalization.")
        
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_nested_cv()
