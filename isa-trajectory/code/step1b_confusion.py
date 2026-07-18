import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def extract_layer0_residuals(model, tokenizer, prompts):
    print(f"Extracting layer 0 residuals for {len(prompts)} prompts...")
    
    if "Qwen" in model.config.architectures[0] or "Llama" in model.config.architectures[0]:
        layer0 = model.model.layers[0]
    elif "Phi" in model.config.architectures[0]:
        layer0 = model.model.layers[0]
        
    cache = []
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() == 3: 
            val = hidden[0, -1, :].detach().cpu()
        else: 
            val = hidden[-1, :].detach().cpu()
        cache.append(val)
        
    handle = layer0.register_forward_hook(hook)
        
    labels = []
    for p in tqdm(prompts, desc="Forward passes"):
        labels.append(CAT_TO_ID[p["task_type"]])
        tokens = tokenizer(p["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**tokens)
            
    handle.remove()
    return torch.stack(cache).numpy(), np.array(labels)

def pytorch_ridge_classifier_preds(X_train, y_train, X_test, num_classes=6, l2_reg=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    Y_train_oh = torch.nn.functional.one_hot(y_train_t, num_classes=num_classes).float()
    I = torch.eye(X_train_t.shape[1]).to(device)
    W = torch.linalg.solve(X_train_t.T @ X_train_t + l2_reg * I, X_train_t.T @ Y_train_oh)
    
    preds_test = (X_test_t @ W).argmax(dim=1)
    return preds_test.cpu().numpy()

def main():
    train_prompts, test_prompts = load_data()
    
    os.makedirs("../outputs/probing", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (m_name, m_path) in enumerate(MODELS.items()):
        print(f"\nProbing {m_name}")
        tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            m_path, 
            torch_dtype=torch.float16 if "Qwen" in m_name or "Llama" in m_name else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        train_X, train_y = extract_layer0_residuals(model, tokenizer, train_prompts)
        test_X, test_y = extract_layer0_residuals(model, tokenizer, test_prompts)
        
        del model
        torch.cuda.empty_cache()
        
        preds = pytorch_ridge_classifier_preds(train_X, train_y, test_X)
        cm = confusion_matrix(test_y, preds, labels=range(6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=CATEGORIES, yticklabels=CATEGORIES)
        axes[i].set_title(f"{m_name} Layer 0")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig("../outputs/probing/confusion_matrix.png")
    print("Saved outputs/probing/confusion_matrix.png")

if __name__ == "__main__":
    main()
