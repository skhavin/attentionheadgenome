"""
visualize_kv_geometry.py
────────────────────────
Visualizes the 3D PCA projection of Key (K) vectors for the four HeadGenome 
canonical head types, demonstrating their distinct geometric representations 
of the exact same token sequence.
"""

import os
os.environ["HF_HOME"] = r"d:\.cache\huggingface"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "geometry")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "openai-community/gpt2-medium"
# Optimal canonical heads from canonical_labels.json
HEADS = {
    "Sink":      (5, 11),  # (layer, head_idx)
    "Local":     (23, 5),
    "Retrieval": (15, 8),
    "Induction": (9, 3)
}

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "A very quick brown fox often jumps over a very lazy dog. "
    "Why does the quick brown fox jump over the lazy dog? "
    "Because the quick brown fox loves to jump over the lazy dog! "
    "Scientists observe that the quick brown fox and the lazy dog "
    "are frequently found together in typing exercises."
)

def get_token_color(idx, token_str):
    t = token_str.strip().lower()
    if idx == 0:
        return "#FF0000", "First Token"  # Red
    elif t in ['.', ',', '!', '?', ';', ':', 'Ġ.', 'Ġ,', 'Ġ!', 'Ġ?']:
        return "#A855F7", "Punctuation"  # Purple
    elif t in ['the', 'a', 'is', 'of', 'and', 'to', 'in', 'it', 'for', 'with', 'on', 'as', 'by', 'at', 'are', 'that']:
        return "#4B5563", "Stopwords"    # Gray
    elif t in ['fox', 'dog']:
        return "#06B6D4", "Key Entities (fox/dog)"  # Cyan
    elif t in ['quick', 'brown', 'lazy', 'jumps', 'jump']:
        return "#10B981", "Attributes/Actions" # Green
    else:
        return "#3B82F6", "Other Words"  # Blue

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([idx]) for idx in input_ids]
    
    print(f"Prompt length: {len(tokens)} tokens")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    
    past_kv = outputs.past_key_values # tuple of length n_layers
    # Each element: (key, value) of shape (batch, num_heads, seq_len, head_dim)

    fig = plt.figure(figsize=(20, 16), facecolor="#0F0F0F")
    fig.suptitle("KV Cache Geometry (PCA Projection of Key Vectors)\nHow Different Head Types See the Same Text", 
                 color="white", fontsize=24, fontweight="bold", y=0.95)

    plot_idx = 1
    for name, (layer, head) in HEADS.items():
        print(f"Processing {name} head (L{layer} H{head})...")
        
        # Extract Key vectors: shape (seq_len, head_dim)
        # NOTE: For GQA models (Qwen, Llama), the head index in past_kv is the 
        # KV-group index, not the query-head index. For GPT-2 (MHA), they are 1:1.
        # If extending to GQA, map `head` -> `head // (num_q_heads // num_kv_heads)`.
        K = past_kv[layer][0][0, head, :, :].cpu().numpy().astype(np.float32)
        
        # Reduce to 3D via PCA
        pca = PCA(n_components=3, random_state=42)
        K_3d = pca.fit_transform(K)
        
        ax = fig.add_subplot(2, 2, plot_idx, projection='3d')
        ax.set_facecolor("#1A1A2E")
        ax.xaxis.set_pane_color((0.1, 0.1, 0.18, 1.0))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.18, 1.0))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.18, 1.0))
        ax.grid(True, color="#333355", linestyle="--", linewidth=0.5)
        
        ax.xaxis.line.set_color("#333355")
        ax.yaxis.line.set_color("#333355")
        ax.zaxis.line.set_color("#333355")
        ax.tick_params(colors="white")

        # Scatter plot
        legend_handles = {}
        for i in range(len(tokens)):
            color, label = get_token_color(i, tokens[i])
            # Draw point
            ax.scatter(K_3d[i, 0], K_3d[i, 1], K_3d[i, 2], c=color, s=50, alpha=0.8, edgecolors="white", linewidth=0.5)
            
            # Connect consecutive tokens with a thin line to show the sequence path
            if i > 0:
                ax.plot([K_3d[i-1, 0], K_3d[i, 0]], 
                        [K_3d[i-1, 1], K_3d[i, 1]], 
                        [K_3d[i-1, 2], K_3d[i, 2]], 
                        color="#4B5563", alpha=0.3, linewidth=1)
                
            if label not in legend_handles:
                legend_handles[label] = color
                
        # Annotate specific tokens to make it readable
        # Only annotate punctuation, first token, and entities to avoid clutter
        for i in range(len(tokens)):
            color, label = get_token_color(i, tokens[i])
            if label in ["First Token", "Punctuation", "Key Entities (fox/dog)"]:
                # add slight offset
                ax.text(K_3d[i, 0], K_3d[i, 1], K_3d[i, 2]+0.02, tokens[i].strip(), 
                        color="white", fontsize=8, alpha=0.8)

        var_explained = sum(pca.explained_variance_ratio_) * 100
        ax.set_title(f"{name} Head (L{layer} H{head})\nExplained Var: {var_explained:.1f}%", 
                     color="white", fontsize=16, pad=10)
        
        # Add legend only to the first plot
        if plot_idx == 1:
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=c, label=l) for l, c in legend_handles.items()]
            ax.legend(handles=patches, facecolor="#111122", labelcolor="white", loc="upper left")

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, "figure12_kv_geometry.png")
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved visualization to {out_path}")

if __name__ == "__main__":
    main()
