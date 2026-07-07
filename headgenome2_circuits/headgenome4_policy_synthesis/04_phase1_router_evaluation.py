import os
import torch
import random
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODELS = {
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B",
    # We can run smaller models for faster benchmarking
}

def evaluate_hellaswag(model, tokenizer, num_samples=100, head_mask=None):
    """
    Evaluates the model on a subset of HellaSwag.
    head_mask: Tensor of shape (num_layers, num_heads) with 1s and 0s.
    """
    dataset = load_dataset("hellaswag", split="validation")
    # Take a random subset to save time
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for row in tqdm(dataset, desc="Evaluating HellaSwag"):
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        
        best_score = float("-inf")
        best_pred = -1
        
        for i, ending in enumerate(endings):
            prompt = ctx + " " + ending
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                # We use head_mask to ablate specific heads
                outputs = model(**inputs, head_mask=head_mask)
                logits = outputs.logits[0, :-1, :]
                target_ids = inputs.input_ids[0, 1:]
                
                # Calculate log probability of the ending
                # We only care about the tokens in the ending
                ending_tokens = tokenizer(ending, return_tensors="pt").input_ids[0]
                if len(ending_tokens) == 0: continue
                # To be precise, we should only sum the log probs of the ending tokens.
                # For simplicity in this benchmark, we'll just sum log probs of the whole sequence.
                # Since the context is the same, the difference is just the ending.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                seq_log_prob = torch.gather(log_probs, 1, target_ids.unsqueeze(1)).squeeze(1).sum().item()
                
                # Normalize by length to prevent bias towards short endings
                seq_log_prob /= len(target_ids)
                
            if seq_log_prob > best_score:
                best_score = seq_log_prob
                best_pred = i
                
        if best_pred == label:
            correct += 1
        total += 1
        
    return correct / total

def get_universal_algorithm_mask(model_key, n_layers, n_heads):
    """
    Applies the Universal Mechanistic Algorithm rules to decide which heads to KEEP.
    We will create a mask that KEEPS Induction and Retrieval heads, and drops Sink/Local,
    or vice versa to test ablation impact.
    """
    path = f"outputs/phase1/component_attribution_{model_key}.csv"
    if not os.path.exists(path):
        print(f"No attribution data for {model_key}. Cannot build router mask.")
        return None
        
    df = pd.read_csv(path)
    
    # Initialize full mask (keep all)
    mask = torch.ones((n_layers, n_heads), device="cuda")
    
    # Identify heads based on algorithm
    induction_heads = []
    retrieval_heads = []
    
    for _, row in df.iterrows():
        l, h = int(row['layer']), int(row['head'])
        embed_k_pct = row['k_embed_contrib'] / (row['total_score'] + 1e-6)
        q_layer = row['top_q_layer']
        k_layer = row['top_k_layer']
        
        # Rule 3: Retrieval (moderate embed K, q > k)
        if embed_k_pct > 0.01 and q_layer > k_layer:
            retrieval_heads.append((l, h))
        # Rule 4: Induction (low embed K, q > k)
        elif embed_k_pct <= 0.01 and q_layer > k_layer:
            induction_heads.append((l, h))
            
    print(f"Router identified {len(induction_heads)} Induction and {len(retrieval_heads)} Retrieval heads.")
    
    # Let's create an ablation mask that ZEROS OUT all Induction and Retrieval heads.
    # If they are crucial for reasoning, HellaSwag should tank.
    ablated_mask = torch.ones((n_layers, n_heads), device="cuda")
    for l, h in induction_heads + retrieval_heads:
        if l < n_layers and h < n_heads:
            ablated_mask[l, h] = 0.0
            
    num_ablated = len(induction_heads) + len(retrieval_heads)
    return ablated_mask, num_ablated

def run_benchmarks():
    print("=== HellaSwag Ablation Benchmarks ===")
    
    for model_key, model_id in MODELS.items():
        print(f"\nEvaluating {model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
        model.eval()
        
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        
        # 1. Full Model Baseline
        print("  Running Full Model Baseline...")
        full_acc = evaluate_hellaswag(model, tokenizer, num_samples=100, head_mask=None)
        print(f"  Full Model Accuracy: {full_acc*100:.1f}%")
        
        # 2. Router Ablation (Zero out Induction/Retrieval)
        router_mask, num_ablated = get_universal_algorithm_mask(model_key, n_layers, n_heads)
        if router_mask is not None and num_ablated > 0:
            print(f"  Running Router Ablation (Zeroing {num_ablated} Induction/Retrieval heads)...")
            router_acc = evaluate_hellaswag(model, tokenizer, num_samples=100, head_mask=router_mask)
            print(f"  Router Ablation Accuracy: {router_acc*100:.1f}%")
            
            # 3. Random Baseline (Zero out same number of random heads)
            print(f"  Running Random Ablation (Zeroing {num_ablated} random heads)...")
            rand_mask = torch.ones((n_layers, n_heads), device="cuda")
            all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
            random.seed(42)
            rand_ablated = random.sample(all_heads, num_ablated)
            for l, h in rand_ablated:
                rand_mask[l, h] = 0.0
                
            rand_acc = evaluate_hellaswag(model, tokenizer, num_samples=100, head_mask=rand_mask)
            print(f"  Random Ablation Accuracy: {rand_acc*100:.1f}%")
            
            print(f"  Drop from Router Ablation: {full_acc - router_acc:.3f}")
            print(f"  Drop from Random Ablation: {full_acc - rand_acc:.3f}")
            if (full_acc - router_acc) > (full_acc - rand_acc):
                print("  [SUCCESS] Router correctly identified causal reasoning heads! Ablating them hurt more than random.")
            else:
                print("  [FAILURE] Ablating router heads hurt less than random.")

if __name__ == "__main__":
    run_benchmarks()
