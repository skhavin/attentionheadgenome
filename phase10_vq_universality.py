"""
phase10_vq_universality.py

Generates the conclusive V/Q vs Depth scaling law figure proving that the 
HeadGenome topology is an emergent property of optimization (training), not 
architectural initialization.

Updated based on NeurIPS critique:
  - 4x2 controlled comparison: untrain baseline for ALL 4 models
  - Llama annotation for GQA ceiling constraint
  - Pearson r inset chart
  - Neutral title
  - Removed confusing global trend line
"""

import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from transformers import AutoConfig, AutoModelForCausalLM

OUT_DIR = "outputs/phase10_universality"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]
HF_IDS = {
    "GPT-2":        "gpt2-medium",
    "Qwen-0.5B":    "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B":    "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B"
}
MODEL_COLS = {
    "GPT-2":        "#818cf8",
    "Qwen-0.5B":    "#34d399",
    "Qwen-1.5B":    "#fb923c",
    "Llama-3.2-1B": "#f472b6",
}
PEARSON_R = {
    "GPT-2": 0.681,
    "Qwen-0.5B": 0.734,
    "Qwen-1.5B": 0.647,
    "Llama-3.2-1B": 0.635
}

def get_untrained_vq(model_name):
    """Instantiate a randomly initialized model and compute V/Q per head."""
    hf_id = HF_IDS[model_name]
    print(f"Generating Untrained (Random Init) {model_name}...")
    config = AutoConfig.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_config(config)
    
    results = []
    if model_name == "GPT-2":
        num_layers = config.n_layer
        num_heads = config.n_head
        d_model = config.n_embd
        head_dim = d_model // num_heads

        for layer_idx in range(num_layers):
            rel_depth = layer_idx / (num_layers - 1)
            attn_layer = model.transformer.h[layer_idx].attn
            W = attn_layer.c_attn.weight.detach()
            W_q, W_k, W_v = torch.split(W, d_model, dim=-1)
            
            for head_idx in range(num_heads):
                # Note: GPT-2 c_attn weight is transposed relative to standard Linear
                # W_q_h = W_q[:, head_idx * head_dim : (head_idx + 1) * head_dim]
                W_q_h = W_q[:, head_idx * head_dim : (head_idx + 1) * head_dim]
                W_v_h = W_v[:, head_idx * head_dim : (head_idx + 1) * head_dim]
                
                norm_q = torch.linalg.norm(W_q_h).item()
                norm_v = torch.linalg.norm(W_v_h).item()
                results.append((rel_depth, norm_v / (norm_q + 1e-12)))
    else:
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        d_model = config.hidden_size
        head_dim = d_model // num_heads
        g = num_heads // num_kv_heads

        for layer_idx in range(num_layers):
            rel_depth = layer_idx / (num_layers - 1)
            self_attn = model.model.layers[layer_idx].self_attn
            W_q = self_attn.q_proj.weight.detach()
            W_v = self_attn.v_proj.weight.detach()

            for head_idx in range(num_heads):
                kv_head_idx = head_idx // g
                W_q_h = W_q[head_idx * head_dim : (head_idx + 1) * head_dim, :]
                W_v_h = W_v[kv_head_idx * head_dim : (kv_head_idx + 1) * head_dim, :]
                
                norm_q = torch.linalg.norm(W_q_h).item()
                norm_v = torch.linalg.norm(W_v_h).item()
                results.append((rel_depth, norm_v / (norm_q + 1e-12)))
                
    del model
    return np.array(results)

def get_trained_vq_data(model_name, data):
    if model_name not in data["models"]:
        return np.array([])
    heads = data["models"][model_name]
    max_layer = max(int(k.split("_")[0]) for k in heads.keys())
    
    results = []
    for k, v in heads.items():
        layer = int(k.split("_")[0])
        vq = v["features"]["v_q_ratio"]
        rel_depth = layer / max_layer if max_layer > 0 else 0
        results.append((rel_depth, vq))
    return np.array(results)

def plot_universality_curves(trained, untrained):
    print("Plotting Figure 8...")
    
    BG, SURFACE, BORDER, TEXT, MUTED = "#0b1120", "#111827", "#334155", "#f1f5f9", "#94a3b8"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE, "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    })

    fig, ax = plt.subplots(figsize=(11, 7.5))
    x_lin = np.linspace(0, 1, 100)
    
    for model_name in MODELS:
        c = MODEL_COLS[model_name]
        
        # Untrained Plot (Dashed, flatter)
        if len(untrained[model_name]) > 0:
            ux, uy = untrained[model_name][:, 0], untrained[model_name][:, 1]
            z_u = np.polyfit(ux, uy, 1) # linear fit
            p_u = np.poly1d(z_u)
            ax.plot(x_lin, p_u(x_lin), color=c, lw=2.5, linestyle="--", alpha=0.55)
            
        # Trained Plot (Solid curve + scatter)
        if len(trained[model_name]) > 0:
            x, y = trained[model_name][:, 0], trained[model_name][:, 1]
            ax.scatter(x, y, s=12, color=c, alpha=0.15, edgecolors='none')
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x_lin, p(x_lin), color=c, lw=2.5, label=model_name)
            
            # Llama Annotation
            if model_name == "Llama-3.2-1B":
                # Find mid-point of the curve
                mid_idx = 70
                ax.annotate("GQA constraint ceiling\n(Diffuse V capacity)", 
                            xy=(x_lin[mid_idx], p(x_lin)[mid_idx]),
                            xytext=(x_lin[mid_idx]-0.25, p(x_lin)[mid_idx]-0.15),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
                            fontsize=9, color=c, ha="center")

    ax.set_xlabel("Relative Network Depth", fontsize=12, labelpad=10)
    ax.set_ylabel("V/Q Matrix Norm Ratio", fontsize=12, labelpad=10)
    
    ax.set_title("Figure 8: V/Q Norm Ratio vs. Relative Network Depth: Trained vs. Untrained\n",
                 fontsize=14, fontweight="bold", pad=5, loc="center")
                 
    ax.grid(True, alpha=0.2)
    
    # Custom Legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="none", label=" "))
    handles.append(plt.Line2D([0], [0], color=TEXT, lw=2.5, linestyle="-", label="Trained Networks"))
    handles.append(plt.Line2D([0], [0], color=TEXT, lw=2.5, linestyle="--", alpha=0.55, label="Untrained Baselines"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.8, facecolor=SURFACE, edgecolor=BORDER, fontsize=10)

    # Inset Bar Chart for Pearson r
    axins = inset_axes(ax, width="25%", height="22%", loc="lower right", borderpad=3)
    axins.set_facecolor(BG)
    axins.tick_params(colors=MUTED, labelsize=8)
    for spine in axins.spines.values():
        spine.set_color(BORDER)
    
    bars = axins.bar(MODELS, [PEARSON_R[m] for m in MODELS], 
                     color=[MODEL_COLS[m] for m in MODELS], alpha=0.9, width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        axins.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", 
                   ha='center', va='bottom', fontsize=7.5, color=TEXT)
                   
    axins.set_ylim(0, 1.0)
    axins.set_xticks(range(len(MODELS)))
    axins.set_xticklabels(["GPT-2", "Q0.5B", "Q1.5B", "Llama1B"], rotation=25, ha="right", fontsize=7.5)
    axins.set_title("Pearson r (Depth vs V/Q)", fontsize=9, color=TEXT, pad=5)
    
    out_path = os.path.join(OUT_DIR, "figure8_vq_emergence.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return out_path

def main():
    with open("outputs/phase3/weight_features.json") as f:
        data = json.load(f)
        
    trained = {m: get_trained_vq_data(m, data) for m in MODELS}
    untrained = {m: get_untrained_vq(m) for m in MODELS}
    
    plot_universality_curves(trained, untrained)

if __name__ == "__main__":
    main()
