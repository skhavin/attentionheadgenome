import pandas as pd
import numpy as np
import os
import sys

# Import the schema to get Bin-3 features
import importlib.util
spec = importlib.util.spec_from_file_location(
    "triage_and_schema", 
    os.path.join(os.path.dirname(__file__), "01_triage_and_schema.py")
)
triage_and_schema = importlib.util.module_from_spec(spec)
sys.modules["triage_and_schema"] = triage_and_schema
spec.loader.exec_module(triage_and_schema)
from triage_and_schema import assert_no_bin3_leak

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "full_genome_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Problem 1: Confirm no Bin-3 Leakage
    print("\n--- 1. Confirming No Bin-3 Leakage ---")
    try:
        assert_no_bin3_leak(df)
        print("[PASS] No Bin-3 features found in the dataset.")
    except AssertionError as e:
        print(f"[FAIL] Bin-3 leak detected: {e}")
        return

    # Prepare features
    exclude_cols = ['model_name', 'model_id', 'canonical_label']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Problem 2 & 3: Per-architecture Z-scores & std check
    print("\n--- 2. & 3. Per-Architecture Analysis & Local Std Check ---")
    
    models = df['model_name'].unique()
    classes = [c for c in df['canonical_label'].unique() if c != 'local']
    
    # Store top features for overlap checking
    # top_features[class][model] = set(top_features)
    top_features_by_class = {c: {} for c in classes}
    
    for model in models:
        print(f"\n==================== Architecture: {model} ====================")
        df_model = df[df['model_name'] == model]
        
        # Get Local class stats
        df_local = df_model[df_model['canonical_label'] == 'local']
        if len(df_local) == 0:
            print(f"  No 'local' heads found for {model}. Skipping.")
            continue
            
        local_means = df_local[features].mean()
        local_stds = df_local[features].std()
        
        # Flag tiny stds to avoid Z-score inflation
        tiny_std_threshold = 1e-5
        tiny_stds = local_stds[local_stds < tiny_std_threshold]
        
        for cls in classes:
            df_cls = df_model[df_model['canonical_label'] == cls]
            if len(df_cls) == 0:
                print(f"\n  [Class: {cls}] No heads found. Skipping.")
                top_features_by_class[cls][model] = []
                continue
                
            cls_means = df_cls[features].mean()
            
            # Compute Z-score vs Local
            # z = (mean_cls - mean_local) / local_std
            # We clip the denominator to avoid infinity
            z_scores = (cls_means - local_means) / local_stds.clip(lower=tiny_std_threshold)
            z_scores_abs = z_scores.abs().sort_values(ascending=False)
            
            print(f"\n  [Class: {cls}] Top 10 Divergent Features vs Local (by Z-score):")
            
            top_10 = []
            for feat, z_abs in z_scores_abs.head(10).items():
                z = z_scores[feat]
                l_std = local_stds[feat]
                
                # Check for tiny std inflation
                inflation_warning = " ⚠️ [TINY STD INFLATION]" if l_std < tiny_std_threshold else ""
                
                print(f"    - {feat:<30} Z={z:>7.2f}  (local_std={l_std:.2e}){inflation_warning}")
                
                # Only consider it a "real" top feature if it doesn't have a tiny std
                if l_std >= tiny_std_threshold:
                    top_10.append(feat)
            
            top_features_by_class[cls][model] = set(top_10)

    print("\n--- 4. Cross-Architecture Overlap ---")
    for cls in classes:
        print(f"\n  [Class: {cls}] Consistent Top Features Across ALL Architectures:")
        sets = list(top_features_by_class[cls].values())
        
        # We only care about models that actually had this class
        valid_sets = [s for s in sets if len(s) > 0]
        
        if len(valid_sets) < 2:
            print(f"    Not enough architectures have heads of class '{cls}' to check overlap.")
            continue
            
        overlap = set.intersection(*valid_sets)
        if overlap:
            for feat in overlap:
                print(f"    * {feat}")
        else:
            print("    None. (No feature consistently appeared in the Top 10 across all tested architectures).")

if __name__ == "__main__":
    main()
