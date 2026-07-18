import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def pytorch_ridge_classifier(X_train, y_train, X_test, y_test, num_classes=6, l2_reg=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    Y_train_oh = torch.nn.functional.one_hot(y_train_t, num_classes=num_classes).float()
    
    # Ridge regression formula: W = (X^T X + lambda I)^-1 X^T Y
    I = torch.eye(X_train_t.shape[1]).to(device)
    W = torch.linalg.solve(X_train_t.T @ X_train_t + l2_reg * I, X_train_t.T @ Y_train_oh)
    
    preds_test = (X_test_t @ W).argmax(dim=1)
    acc = (preds_test == y_test_t).float().mean().item()
    return acc, preds_test.cpu().numpy()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
    "phi-1_5": "microsoft/phi-1_5"
}

CATEGORIES = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

def load_data():
    with open("../outputs/dataset/trajectory_mapping.json", "r", encoding="utf-8") as f:
        train_prompts = json.load(f)
    with open("../outputs/dataset/trajectory_validation.json", "r", encoding="utf-8") as f:
        test_prompts = json.load(f)
    return train_prompts, test_prompts

def extract_residuals(model, tokenizer, prompts):
    print(f"Extracting residuals for {len(prompts)} prompts...")
    
    # Setup hooks based on model type
    if "Qwen" in model.config.architectures[0] or "Llama" in model.config.architectures[0]:
        layers = model.model.layers
    elif "Phi" in model.config.architectures[0]:
        layers = model.model.layers
    else:
        raise ValueError("Unknown architecture")
        
    n_layers = len(layers)
    D = model.config.hidden_size
    
    cache = {l: [] for l in range(n_layers)}
    def get_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3: 
                val = hidden[0, -1, :].detach().cpu()
            else: 
                val = hidden[-1, :].detach().cpu()
            cache[layer_idx].append(val)
        return hook
        
    handles = []
    for l in range(n_layers):
        handles.append(layers[l].register_forward_hook(get_hook(l)))
        
    labels = []
    for p in tqdm(prompts, desc="Forward passes"):
        labels.append(CAT_TO_ID[p["task_type"]])
        tokens = tokenizer(p["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**tokens)
            
    for h in handles:
        h.remove()
        
    # Stack: [N, L, D]
    N = len(prompts)
    trajectories = torch.zeros((N, n_layers, D))
    for l in range(n_layers):
        trajectories[:, l, :] = torch.stack(cache[l])
        
    return trajectories.numpy(), np.array(labels)

def run_probing(m_name, m_path, train_prompts, test_prompts):
    print(f"\n{'='*50}\nProbing {m_name}\n{'='*50}")
    
    tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        m_path, 
        torch_dtype=torch.float16 if "Qwen" in m_name or "Llama" in m_name else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    train_X, train_y = extract_residuals(model, tokenizer, train_prompts)
    test_X, test_y = extract_residuals(model, tokenizer, test_prompts)
    
    # Calculate lengths
    train_lengths = np.array([len(tokenizer.encode(p["prompt"])) for p in train_prompts]).reshape(-1, 1)
    test_lengths = np.array([len(tokenizer.encode(p["prompt"])) for p in test_prompts]).reshape(-1, 1)
    
    scaler_len = StandardScaler()
    train_lengths = scaler_len.fit_transform(train_lengths)
    test_lengths = scaler_len.transform(test_lengths)
    
    length_only_acc, _ = pytorch_ridge_classifier(train_lengths, train_y, test_lengths, test_y)
    print(f"Length-Only Baseline Accuracy: {length_only_acc:.3f}")
    
    # Free VRAM
    del model
    torch.cuda.empty_cache()
    
    n_layers = train_X.shape[1]
    
    real_accuracies = []
    shuffle_95th_percentiles = []
    
    # Store per-layer metrics
    layer_metrics = []
    
    for l in range(n_layers):
        X_train_l = train_X[:, l, :]
        X_test_l = test_X[:, l, :]
        
        scaler = StandardScaler()
        X_train_l = scaler.fit_transform(X_train_l)
        X_test_l = scaler.transform(X_test_l)
        
        # True Probe
        real_acc, preds = pytorch_ridge_classifier(X_train_l, train_y, X_test_l, test_y)
        real_accuracies.append(real_acc)
        
        # Shuffled Probes (100 shuffles)
        shuffled_accs = []
        for _ in range(100):
            shuffled_y = np.random.permutation(train_y)
            shuff_acc, _ = pytorch_ridge_classifier(X_train_l, shuffled_y, X_test_l, test_y)
            shuffled_accs.append(shuff_acc)
            
        p95 = np.percentile(shuffled_accs, 95)
        shuffle_95th_percentiles.append(p95)
        
        layer_metrics.append({
            "layer": l,
            "real_accuracy": real_acc,
            "shuffle_95th_percentile": p95,
            "is_significant": bool(real_acc > p95),
            "length_baseline_accuracy": length_only_acc
        })
        
        if l == 0:
            cm = confusion_matrix(test_y, preds, labels=range(6))
            cat_accs = cm.diagonal() / cm.sum(axis=1)
            print(f"Layer 00 Per-Category Accuracies:")
            cats = ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]
            for c_name, c_acc in zip(cats, cat_accs):
                print(f"  - {c_name}: {c_acc:.3f}")
                
        print(f"Layer {l:02d} | Real Acc: {real_acc:.3f} | Shuffle 95th: {p95:.3f} | Significant: {real_acc > p95}")
        
    return layer_metrics

def main():
    train_prompts, test_prompts = load_data()
    
    os.makedirs("../outputs/probing", exist_ok=True)
    all_results = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, (m_name, m_path) in enumerate(MODELS.items()):
        layer_metrics = run_probing(m_name, m_path, train_prompts, test_prompts)
        all_results[m_name] = layer_metrics
        
        # Plotting
        plt.subplot(1, 3, i+1)
        layers = [m["layer"] for m in layer_metrics]
        real_accs = [m["real_accuracy"] for m in layer_metrics]
        p95s = [m["shuffle_95th_percentile"] for m in layer_metrics]
        
        # Calculate Wilson CIs
        ci_lower = []
        ci_upper = []
        for acc in real_accs:
            count = int(round(acc * 180))
            lower, upper = proportion_confint(count, 180, method='wilson')
            ci_lower.append(lower)
            ci_upper.append(upper)
            
        plt.plot(layers, real_accs, marker='o', label="Real Accuracy", color='blue')
        plt.fill_between(layers, ci_lower, ci_upper, color='blue', alpha=0.2, label="95% CI (Wilson)")
        
        plt.plot(layers, p95s, linestyle='--', label="95th%ile Shuffle", color='red')
        plt.axhline(1/6, color='gray', linestyle=':', label="Random Chance (16.7%)")
        plt.title(f"{m_name}")
        plt.xlabel("Layer")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1.05)
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig("../outputs/probing/probing_results.png", dpi=300)
    
    with open("../outputs/probing/probing_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nProbing complete! Results saved to outputs/probing/")

if __name__ == "__main__":
    main()
