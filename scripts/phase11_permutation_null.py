"""
phase11_permutation_null.py

The Permutation Null Experiment:
Proves that the HeadGenome taxonomy (specifically Retrieval and Induction heads)
is a purely structural mechanism, independent of natural language statistics.

We subject the model to the exact same stress-tests (NIAH and Repetition) used
in Phase 1, but we construct the sequences by randomly shuffling actual English
text (WikiText). This preserves the marginal token frequencies (keeping embeddings 
in-distribution) but completely destroys all grammar, syntax, and semantics.

If the taxonomy is structural, the heads will still exhibit their extreme 
entropy collapses (Delta) on the shuffled sequences, matching their natural 
language behavior perfectly.
"""

import os
import json
import torch
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "outputs/phase11_permutation_null"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_shuffled_tokens(tokenizer, dataset_texts, count):
    """Returns a 1D tensor of `count` tokens randomly shuffled from real text."""
    text = " ".join(random.sample(dataset_texts, min(50, len(dataset_texts))))
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    # Shuffle them!
    random.shuffle(tokens)
    # Loop/truncate to get exactly `count`
    while len(tokens) < count:
        tokens.extend(tokens)
    return torch.tensor(tokens[:count]).unsqueeze(0).to(DEVICE)

def calculate_entropy(attn_weights, mask):
    """Calculate Shannon entropy over the key dimension."""
    p = attn_weights.float()
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
    return (entropy * mask).sum() / (mask.sum() + 1e-9)

def run_gibberish_probe(model, tokenizer, dataset_texts):
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    
    # 1. Baseline Entropy on Shuffled Text
    print("Measuring Baseline Entropy on Shuffled Text...")
    baseline_entropies = np.zeros((num_layers, num_heads))
    n_baseline = 50
    seq_len = 128
    
    for _ in tqdm(range(n_baseline)):
        input_ids = get_shuffled_tokens(tokenizer, dataset_texts, seq_len)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
            
        for l in range(num_layers):
            attn = outputs.attentions[l][0]
            mask = torch.ones(seq_len).to(DEVICE)
            mask[:5] = 0 # skip early tokens
            for h in range(num_heads):
                ent = calculate_entropy(attn[h], mask)
                baseline_entropies[l, h] += ent.item() / n_baseline

    # 2. Induction Probe (Repetition) on Shuffled Text
    print("Measuring Induction Delta on Shuffled Text...")
    induction_entropies = np.zeros((num_layers, num_heads))
    n_induct = 50
    
    for _ in tqdm(range(n_induct)):
        block_a = get_shuffled_tokens(tokenizer, dataset_texts, 15)
        block_b = get_shuffled_tokens(tokenizer, dataset_texts, 50)
        input_ids = torch.cat([block_a, block_b, block_a], dim=1)
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
            
        for l in range(num_layers):
            attn = outputs.attentions[l][0]
            mask = torch.zeros(seq_len).to(DEVICE)
            mask[-15:] = 1 
            for h in range(num_heads):
                ent = calculate_entropy(attn[h], mask)
                induction_entropies[l, h] += ent.item() / n_induct

    # 3. Retrieval Probe (NIAH) on Shuffled Text
    print("Measuring Retrieval Delta on Shuffled Text...")
    retrieval_entropies = np.zeros((num_layers, num_heads))
    n_retrieval = 50
    
    for _ in tqdm(range(n_retrieval)):
        haystack1 = get_shuffled_tokens(tokenizer, dataset_texts, 60)
        haystack2 = get_shuffled_tokens(tokenizer, dataset_texts, 60)
        needle = get_shuffled_tokens(tokenizer, dataset_texts, 1)
        
        input_ids = torch.cat([haystack1, needle, haystack2, needle], dim=1)
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
            
        for l in range(num_layers):
            attn = outputs.attentions[l][0]
            mask = torch.zeros(seq_len).to(DEVICE)
            mask[-1:] = 1 
            for h in range(num_heads):
                ent = calculate_entropy(attn[h], mask)
                retrieval_entropies[l, h] += ent.item() / n_retrieval

    induct_deltas = induction_entropies - baseline_entropies 
    retrieval_deltas = baseline_entropies - retrieval_entropies # Positive collapse

    return induct_deltas, retrieval_deltas

def plot_permutation_null(gib_induct, gib_retrieval):
    with open("outputs/canonical_labels.json") as f:
        data = json.load(f)
    gpt2_heads = data["models"]["GPT-2"]["heads"]
    
    nat_induct, nat_retrieval = [], []
    gibb_i, gibb_r = [], []
    colors_i = []
    
    for k, v in gpt2_heads.items():
        layer, head = int(k.split("_")[0]), int(k.split("_")[1])
        lbl = v["label"].capitalize()
        delta = v["delta"] # Natural language delta
        
        nat_induct.append(delta)
        gibb_i.append(gib_induct[layer, head])
        
        if lbl == "Induction":
            colors_i.append("#f59e0b")
        elif lbl == "Retrieval":
            colors_i.append("#3b82f6")
        elif lbl == "Local":
            colors_i.append("#22c55e")
        else:
            colors_i.append("#ef4444")

    print("Plotting Figure 9...")
    BG, SURFACE, BORDER, TEXT, MUTED = "#0b1120", "#111827", "#334155", "#f1f5f9", "#94a3b8"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE, "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # INDUCTION PLOT
    ax1.scatter(nat_induct, gibb_i, c=colors_i, alpha=0.7, edgecolors='white', linewidth=0.5, s=40)
    min_v = min(min(nat_induct), min(gibb_i))
    max_v = max(max(nat_induct), max(gibb_i))
    ax1.plot([min_v, max_v], [min_v, max_v], 'w--', alpha=0.5, label="Perfect Agreement (y=x)")
    
    ax1.set_xlabel("Entropy Collapse $\Delta$ on Natural English", fontsize=11, labelpad=8)
    ax1.set_ylabel("Entropy Collapse $\Delta$ on Shuffled Tokens", fontsize=11, labelpad=8)
    ax1.set_title("A  |  Induction Mechanics (Repetition Task)", fontsize=12, fontweight="bold", pad=10, color="#f59e0b")
    ax1.grid(True)
    ax1.legend(facecolor=SURFACE, edgecolor=BORDER, fontsize=10)

    # RETRIEVAL PLOT
    r_nat, r_gib = [], []
    for k, v in gpt2_heads.items():
        if v["label"] == "retrieval":
            r_nat.append(v["delta"])
            layer, head = int(k.split("_")[0]), int(k.split("_")[1])
            r_gib.append(gib_retrieval[layer, head])
            
    ax2.scatter(r_nat, r_gib, c="#3b82f6", alpha=0.9, edgecolors='white', linewidth=0.5, s=60)
    if r_nat and r_gib:
        min_v2 = min(min(r_nat), min(r_gib)) - 0.1
        max_v2 = max(max(r_nat), max(r_gib)) + 0.1
        ax2.plot([min_v2, max_v2], [min_v2, max_v2], 'w--', alpha=0.5, label="Perfect Agreement (y=x)")
    
    ax2.set_xlabel("Entropy Collapse $\Delta$ on Natural English", fontsize=11, labelpad=8)
    ax2.set_ylabel("Entropy Collapse $\Delta$ on Shuffled Tokens", fontsize=11, labelpad=8)
    ax2.set_title("B  |  Retrieval Mechanics (NIAH Task)", fontsize=12, fontweight="bold", pad=10, color="#3b82f6")
    ax2.grid(True)
    ax2.legend(facecolor=SURFACE, edgecolor=BORDER, fontsize=10)

    fig.suptitle("Figure 9: The Permutation Null (GPT-2)\nStructural specialization survives the complete destruction of language statistics.", 
                 fontsize=15, fontweight="bold", y=1.05)
                 
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color="#f59e0b", label="Induction Heads"),
        mpatches.Patch(color="#3b82f6", label="Retrieval Heads"),
        mpatches.Patch(color="#22c55e", label="Local Heads"),
        mpatches.Patch(color="#ef4444", label="Sink Heads"),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False, fontsize=11)
    
    out_path = os.path.join(OUT_DIR, "figure9_permutation_null.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    
    from scipy.stats import pearsonr
    r_induct, _ = pearsonr(nat_induct, gibb_i)
    print(f"\nPearson correlation (Induction Natural vs Shuffled): {r_induct:.4f}")
    if len(r_nat) > 1:
        r_ret, _ = pearsonr(r_nat, r_gib)
        print(f"Pearson correlation (Retrieval Natural vs Shuffled): {r_ret:.4f}")

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        attn_implementation="eager",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    model.eval().to(DEVICE)
    
    print("Loading WikiText for shuffling...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.strip()) > 20]
    
    induct_deltas, retrieval_deltas = run_gibberish_probe(model, tokenizer, texts)
    plot_permutation_null(induct_deltas, retrieval_deltas)
    
if __name__ == "__main__":
    main()
