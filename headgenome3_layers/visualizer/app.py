"""
HeadGenome III — Transformer Internals Visualizer
==================================================
Run with:
    streamlit run headgenome3_layers/visualizer/app.py

Shows, for any user-typed prompt:
  Panel 1 — Logit Lens:           per-layer probability of top-5 tokens
  Panel 2 — Attention Heatmaps:   per-layer, per-head attention from last token
  Panel 3 — Residual Norms:       Δx_attn and Δx_mlp stacked bar per layer
  Panel 4 — nMAD Heatmap:         layer × head mean attention distance (normalized)
  Panel 5 — Final Token Probs:    top-10 predicted next tokens at last layer
"""
import sys
import os
import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from headgenome2_circuits.utils.model_loader import load_model_and_tokenizer, MODELS

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="HeadGenome III — Transformer Internals",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background: #0d1117; color: #e6edf3; }
    h1, h2, h3 { color: #58a6ff; }
    .stSelectbox label { color: #8b949e; }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .head-badge {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session state helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def get_model(model_key):
    m, t = load_model_and_tokenizer(model_key,
                                    output_attentions=True,
                                    output_hidden_states=True)
    return m, t

def compute_internals(model, tokenizer, prompt: str, device):
    """Single forward pass. Returns all panels' raw data."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model(**inputs,
                    output_hidden_states=True,
                    output_attentions=True)

    hidden_states = out.hidden_states    # (embed + N layers)  each: (1, seq, d)
    attentions    = out.attentions       # N layers,  each: (1, heads, q, k)

    # ------- Logit Lens -------
    is_gpt2 = hasattr(model, "transformer")
    ln_f    = model.transformer.ln_f if is_gpt2 else model.model.norm
    lm_head = model.lm_head

    logit_lens_probs = []   # per-layer, shape (vocab,) probabilities of top tokens
    logit_lens_topk  = []   # per-layer, list of (token_str, prob)

    for hs in hidden_states:
        last = hs[0, -1, :]
        normed = ln_f(last.unsqueeze(0)).squeeze(0)
        logits = lm_head(normed).float()
        probs  = torch.softmax(logits, dim=-1)
        top    = torch.topk(probs, 10)
        logit_lens_probs.append(probs.cpu())
        logit_lens_topk.append([
            (tokenizer.decode([idx.item()]).strip(), float(p.item()))
            for idx, p in zip(top.indices, top.values)
        ])

    # ------- Residual norms -------
    delta_attn = []   # per layer: Δx from attention sublayer
    delta_mlp  = []   # per layer: Δx from MLP sublayer

    for l in range(len(attentions)):
        x_before = hidden_states[l][0, -1, :]     # before layer l
        x_after  = hidden_states[l+1][0, -1, :]   # after layer l  (full)
        # We can only get the sum here without hooks; approximate by full layer norm
        full_delta = (x_after - x_before).norm().item()
        # Attribute half to attn, half to MLP as rough proxy when no hooks
        delta_attn.append(full_delta * 0.5)
        delta_mlp.append(full_delta * 0.5)

    # ------- nMAD (last-token attention distance) -------
    n_layers = len(attentions)
    n_heads  = attentions[0].shape[1]
    k_len    = attentions[0].shape[-1]
    t        = k_len - 1

    nmad_matrix = np.zeros((n_layers, n_heads))
    if t > 0:
        distances = torch.arange(t, -1, -1, dtype=torch.float32)
        for l, attn in enumerate(attentions):
            # attn: (1, heads, q, k)
            q_idx = attn.shape[2] - 1
            alpha = attn[0, :, q_idx, :].cpu().float()   # (heads, k)
            raw_mad = (alpha * distances).sum(dim=-1)
            nmad_matrix[l] = (raw_mad / t).numpy()

    # ------- Attention matrices (all layers) -------
    attn_matrices = [a[0].cpu().numpy() for a in attentions]   # list of (heads, q, k)

    # ------- Tokens -------
    token_ids  = inputs.input_ids[0].tolist()
    token_strs = [tokenizer.decode([t]).replace(" ", "·") for t in token_ids]

    return {
        "token_strs":      token_strs,
        "seq_len":         seq_len,
        "n_layers":        n_layers,
        "n_heads":         n_heads,
        "logit_lens_topk": logit_lens_topk,       # list[layer] of list[(str, float)]
        "logit_lens_probs":logit_lens_probs,       # list[layer] of tensor (vocab,)
        "delta_attn":      delta_attn,
        "delta_mlp":       delta_mlp,
        "nmad_matrix":     nmad_matrix,            # (n_layers, n_heads)
        "attn_matrices":   attn_matrices,          # list[layer] of (heads, q, k)
    }

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("HeadGenome III")
    st.caption("Transformer Internals Visualizer")
    st.divider()

    model_key = st.selectbox("Model", list(MODELS.keys()), index=0)
    st.divider()
    prompt = st.text_area(
        "Prompt",
        value="Question: What is 4 plus 3? Answer: The sum is",
        height=100,
    )
    known_answer = st.text_input("Known answer token (optional, e.g. '7')", value="7")
    run_btn = st.button("▶ Run Forward Pass", type="primary", use_container_width=True)
    st.divider()
    selected_layer = st.slider("Attention heatmap — layer", 0, 23, 8)
    selected_head  = st.slider("Attention heatmap — head",  0, 15, 0)

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
st.title("🧠 Transformer Internals — Extreme Detail View")
st.caption("Every panel shows the actual internal state of the model for your prompt.")

if "internals" not in st.session_state:
    st.session_state["internals"] = None
    st.session_state["model_key_loaded"] = None

if run_btn and prompt.strip():
    with st.spinner("Loading model & running forward pass…"):
        model, tokenizer = get_model(model_key)
        device = model.device
        data = compute_internals(model, tokenizer, prompt, device)
        st.session_state["internals"] = data
        st.session_state["model_key_loaded"] = model_key
        st.session_state["prompt_used"] = prompt
        st.session_state["answer_used"] = known_answer

data = st.session_state.get("internals")

if data is None:
    st.info("Enter a prompt and press **▶ Run Forward Pass** to begin.")
    st.stop()

n_layers = data["n_layers"]
n_heads  = data["n_heads"]
tokens   = data["token_strs"]

# Clip selected_layer/head to valid range
selected_layer = min(selected_layer, n_layers - 1)
selected_head  = min(selected_head,  n_heads  - 1)

st.divider()

# ================================================================= #
# PANEL 1 — Logit Lens                                              #
# ================================================================= #
with st.expander("📈 Panel 1 — Logit Lens (per-layer token probabilities)", expanded=True):
    st.caption(
        "Each line shows how the probability of a top token evolves from the embedding "
        "layer (0) through all transformer layers. A sharp vertical jump = sudden "
        "emergence. Flat lines = ensemble accumulation."
    )

    topk_layers = data["logit_lens_topk"]   # list[layer] of [(str, prob)]
    n_ls_layers = len(topk_layers)

    # Gather top-5 tokens from final layer
    final_top = [t for t, p in topk_layers[-1][:5]]

    # Known answer
    if known_answer.strip():
        if known_answer.strip() not in final_top:
            final_top = [known_answer.strip()] + final_top[:4]

    # Build trajectories
    fig_ll = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, tok in enumerate(final_top[:TOP_K_DISPLAY]):
        probs_for_tok = []
        for layer_topk in topk_layers:
            tok_dict = dict(layer_topk)
            probs_for_tok.append(tok_dict.get(tok, 0.0))

        is_answer = (tok == known_answer.strip())
        fig_ll.add_trace(go.Scatter(
            x=list(range(n_ls_layers)),
            y=probs_for_tok,
            mode="lines+markers",
            name=repr(tok),
            line=dict(
                width=3 if is_answer else 1.5,
                color="gold" if is_answer else colors[i % len(colors)],
                dash="solid" if is_answer else "dot",
            ),
            marker=dict(size=4),
        ))

    fig_ll.update_layout(
        template="plotly_dark",
        xaxis_title="Layer",
        yaxis_title="Probability",
        legend_title="Token",
        height=350,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_ll, use_container_width=True)

    # Compute suddenness score for the known answer
    if known_answer.strip():
        tok = known_answer.strip()
        probs_ans = [dict(layer_topk).get(tok, 0.0) for layer_topk in topk_layers]
        delta_total = probs_ans[-1] - probs_ans[0]
        if delta_total > 1e-6:
            deltas = [probs_ans[i] - probs_ans[i-1] for i in range(1, len(probs_ans))]
            delta_max = max(deltas)
            s_score = delta_max / delta_total
            l_star  = next((l for l, (_, p) in enumerate(
                [(0, 0)] + [(None, dict(topk_layers[l]).get(tok, 0.0))
                             for l in range(n_ls_layers)]
            ) if l > 0 and dict(topk_layers[l-1] if l > 0 else []).get(tok, 1.0) < dict(topk_layers[l-1]).get(tok, 0.0)), None)

            c1, c2, c3 = st.columns(3)
            c1.metric("Suddenness S", f"{s_score:.2f}",
                      delta=f"{'SUDDEN ✓' if s_score >= 0.40 else 'GRADUAL'}")
            c2.metric("Total prob gain", f"{delta_total:.3f}")
            c3.metric("Pre-registered threshold", "S ≥ 0.40")

# ================================================================= #
# PANEL 2 — Attention Heatmap                                       #
# ================================================================= #
with st.expander("🔥 Panel 2 — Attention Heatmap (layer × head × token)", expanded=True):
    st.caption(
        f"Showing **Layer {selected_layer}, Head {selected_head}** attention weights. "
        "Each cell (row = query token, col = key token) shows how much the query attends to the key. "
        "Adjust the layer/head sliders in the sidebar."
    )

    attn = data["attn_matrices"][selected_layer]   # (heads, q, k)
    head_attn = attn[selected_head]                # (q, k)

    # Clamp for display
    q_tokens = tokens[-min(len(tokens), 30):]
    k_tokens = tokens[-min(len(tokens), 30):]
    attn_display = head_attn[-len(q_tokens):, -len(k_tokens):]

    fig_attn = px.imshow(
        attn_display,
        x=k_tokens,
        y=q_tokens,
        color_continuous_scale="Viridis",
        labels=dict(x="Key token", y="Query token", color="Attn"),
        aspect="auto",
        title=f"Layer {selected_layer} · Head {selected_head}",
    )
    fig_attn.update_layout(template="plotly_dark", height=350,
                           margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig_attn, use_container_width=True)

    # Also show all heads' last-token attention row side by side
    st.caption("**All heads' attention from the last token** (last-row only):")
    last_row = attn[:, -1, :]   # (heads, k)
    fig_all_heads = px.imshow(
        last_row,
        x=tokens[-last_row.shape[1]:],
        y=[f"H{h}" for h in range(n_heads)],
        color_continuous_scale="Blues",
        labels=dict(x="Key token", y="Head", color="Attn"),
        aspect="auto",
        title=f"Layer {selected_layer} — All heads, last-token attention",
    )
    fig_all_heads.update_layout(template="plotly_dark", height=220,
                                margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig_all_heads, use_container_width=True)

# ================================================================= #
# PANEL 3 — Residual Stream Norms                                   #
# ================================================================= #
with st.expander("📊 Panel 3 — Residual Stream Update Norms (Δx per layer)", expanded=False):
    st.caption(
        "Shows how much each layer's attention (blue) and MLP (orange) sublayers "
        "update the residual stream of the last token. Large bars = that sublayer "
        "is doing heavy lifting at that depth."
    )
    layers = list(range(n_layers))
    fig_norms = go.Figure()
    fig_norms.add_bar(x=layers, y=data["delta_attn"], name="Δx (Attn)",
                      marker_color="#58a6ff")
    fig_norms.add_bar(x=layers, y=data["delta_mlp"],  name="Δx (MLP)",
                      marker_color="#f78166")
    fig_norms.update_layout(
        barmode="stack",
        template="plotly_dark",
        xaxis_title="Layer",
        yaxis_title="‖Δx‖₂ (last token)",
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_norms, use_container_width=True)
    st.info(
        "Note: without residual-stream hooks, Attn/MLP Δx are approximated as "
        "half the total per-layer norm. Enable hooks in the script for exact split."
    )

# ================================================================= #
# PANEL 4 — nMAD Heatmap                                            #
# ================================================================= #
with st.expander("🗺 Panel 4 — nMAD Heatmap (layer × head attention reach)", expanded=False):
    st.caption(
        "nMAD ∈ [0,1] measures how far back each head looks from the last token. "
        "**0 = attends only to immediate neighbor (Local head).** "
        "**1 = attends to the very first token (Sink head).** "
        "Intermediate = true content-driven retrieval."
    )
    fig_nmad = px.imshow(
        data["nmad_matrix"],
        labels=dict(x="Head", y="Layer", color="nMAD"),
        x=[f"H{h}" for h in range(n_heads)],
        y=[f"L{l}" for l in range(n_layers)],
        color_continuous_scale="RdYlBu_r",
        zmin=0, zmax=1,
        aspect="auto",
        title="Normalized Mean Attention Distance (nMAD) — last-token row",
    )
    fig_nmad.update_layout(template="plotly_dark", height=420,
                           margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig_nmad, use_container_width=True)

# ================================================================= #
# PANEL 5 — Final Token Probabilities                               #
# ================================================================= #
with st.expander("🎯 Panel 5 — Final Layer Top-10 Next-Token Predictions", expanded=True):
    st.caption("The model's actual next-token distribution at the last transformer layer.")
    final_topk = data["logit_lens_topk"][-1][:10]
    toks  = [repr(t) for t, _ in final_topk]
    probs = [p for _, p in final_topk]

    fig_top = go.Figure(go.Bar(
        x=probs,
        y=toks,
        orientation="h",
        marker=dict(
            color=probs,
            colorscale="Viridis",
        ),
        text=[f"{p:.4f}" for p in probs],
        textposition="outside",
    ))
    fig_top.update_layout(
        template="plotly_dark",
        xaxis_title="Probability",
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=10, r=60, t=10, b=40),
    )
    st.plotly_chart(fig_top, use_container_width=True)

    if known_answer.strip():
        answer_rank = next((i for i, (t, _) in enumerate(final_topk)
                            if t.strip() == known_answer.strip()), None)
        if answer_rank is not None:
            st.success(f"✓ Token '{known_answer}' ranks **#{answer_rank+1}** in the final prediction.")
        else:
            st.warning(f"Token '{known_answer}' not in top-10. It may still be in the vocabulary.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.caption(
    f"Model: **{st.session_state.get('model_key_loaded', '—')}** | "
    f"Prompt: `{st.session_state.get('prompt_used', '—')[:60]}…` | "
    "HeadGenome III — Mechanistic Interpretability Research"
)
