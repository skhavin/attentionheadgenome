import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

MODELS = ["Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B", "Gemma-2B"]

def universal_heuristic_router(row):
    """
    The Universal Mechanistic Algorithm
    Classifies a head's function based purely on its component attribution profile.
    """
    # 1. Local Rule: If it looks very close to the query token, it's local.
    # Note: We don't have seq_len in the CSV, but we can use static mean_distance if available, 
    # or just rely on the other classes. Let's use the attribution features.
    
    embed_k_pct = row['k_embed_contrib'] / (row['total_score'] + 1e-6)
    q_layer = row['top_q_layer']
    k_layer = row['top_k_layer']
    
    # 2. Sink Rule: High reliance on raw embeddings (punctuation/stopwords don't need context)
    # and typically early K layers or BOS token.
    if embed_k_pct > 0.10:
        return "sink"
        
    # 3. Retrieval Rule: Moderate embedding reliance (matching exact token/relation)
    # and Q is built later than K.
    if embed_k_pct > 0.01 and q_layer > k_layer:
        return "retrieval"
        
    # 4. Induction Rule: Deep queries matching mid-level keys, with ZERO reliance on raw embeddings
    # because it matches abstract patterns, not raw text.
    if embed_k_pct <= 0.01 and q_layer > k_layer:
        return "induction"
        
    # Default fallback
    return "local"

def evaluate_algorithm():
    print("=== Universal Mechanistic Algorithm Evaluation ===")
    
    all_y_true = []
    all_y_pred = []
    
    for model in MODELS:
        path = f"outputs/phase1/component_attribution_{model}.csv"
        if not os.path.exists(path):
            print(f"Skipping {model} - missing attribution data.")
            continue
            
        df = pd.read_csv(path)
        
        y_true = df['label'].values
        y_pred = df.apply(universal_heuristic_router, axis=1).values
        
        acc = accuracy_score(y_true, y_pred)
        print(f"\nModel: {model} (Accuracy: {acc*100:.1f}%)")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
    print("\n=== Pooled Cross-Architecture Evaluation ===")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_algorithm()
