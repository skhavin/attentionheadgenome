# Load saved attention tensors and plot heatmaps for each head/layer.
# Output: PNG heatmap images in outputs/phase0/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import PHASE0_DIR

def plot_attention(attention_matrix, tokens, title, save_path):
    """Plot one attention head as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(attention_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticklabels(tokens, fontsize=6)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending)")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    heatmap_dir = os.path.join(PHASE0_DIR, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Just plot sentence 0, layers 0 and 5, all heads — enough to see patterns
    data = torch.load(os.path.join(PHASE0_DIR, "attention_0.pt"), weights_only=False)
    sentence = data["sentence"]
    attentions = data["attentions"]
    tokens = tokenizer.tokenize(sentence)
    tokens = ["[BOS]"] + tokens  # GPT-2 doesn't add BOS but attention dim matches input_ids

    # Trim token labels to match attention matrix size
    seq_len = attentions[0].shape[-1]
    tokens = tokens[:seq_len]

    for layer_idx in [0, 5, 11]:  # first, middle, last layer of gpt2-small (12 layers)
        if layer_idx >= len(attentions):
            continue
        attn = attentions[layer_idx][0]  # (heads, seq, seq)
        for head_idx in range(min(4, attn.shape[0])):  # first 4 heads
            matrix = attn[head_idx].numpy()
            title = f"Layer {layer_idx}, Head {head_idx}"
            path = os.path.join(heatmap_dir, f"L{layer_idx}_H{head_idx}.png")
            plot_attention(matrix, tokens, title, path)
            print(f"Saved {path}")

    print("Done! Check outputs/phase0/heatmaps/")

if __name__ == "__main__":
    main()
