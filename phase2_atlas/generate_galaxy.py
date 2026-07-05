import json
import plotly.graph_objects as go
import os

MODELS = [
    "gpt2-medium",
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Llama-3.2-1B"
]

COLOR_MAP = {
    "Sink": "#A855F7",       # Purple
    "Local": "#10B981",      # Green
    "Retrieval": "#3B82F6",  # Blue
    "Induction": "#F59E0B"   # Orange
}

def load_data(model_name):
    path = f"outputs/phase2_atlas/{model_name}_head_atlas.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)["heads"]

def create_hover_text(head_data):
    # This generates the "Attention DNA" barcode in the tooltip
    # Extract properties
    geom = head_data.get("attention_geometry", {})
    ent = head_data.get("entropy_profile", {})
    gram = head_data.get("grammar_profile", {})
    
    hover = f"<b>{head_data['model']} - L{head_data['layer']}H{head_data['head']} ({head_data['class_label']})</b><br><br>"
    hover += f"<b>Attention DNA</b><br>"
    hover += f"Distance: {geom.get('mean_distance', 0):.1f} | V/Q Ratio: {head_data.get('vq_ratio', 0):.2f}<br>"
    hover += f"Entropy: {ent.get('match_entropy', 0):.2f} | Output Norm: {head_data.get('mean_output_norm', 0):.2f}<br><br>"
    
    if gram:
        hover += f"<b>Grammar Allocation</b><br>"
        # Add small text bars
        nsubj = gram.get("nsubj", 0) * 100
        punct = gram.get("punct", 0) * 100
        case = gram.get("case", 0) * 100
        other = gram.get("other", 0) * 100
        hover += f"Nsubj: {nsubj:.1f}% | Punct: {punct:.1f}%<br>"
        hover += f"Case: {case:.1f}% | Other: {other:.1f}%<br>"
    
    return hover

def build_dashboard():
    fig = go.Figure()
    
    traces = []
    
    for i, model in enumerate(MODELS):
        data = load_data(model)
        if not data:
            print(f"Missing data for {model}")
            continue
            
        xs, ys, zs, colors, texts = [], [], [], [], []
        
        for head_id, meta in data.items():
            geom = meta.get("attention_geometry", {})
            ent = meta.get("entropy_profile", {})
            
            x = geom.get("mean_distance", 0)
            y = -ent.get("match_entropy", 0) # Negative entropy for collapse (higher is collapsed)
            z = meta.get("vq_ratio", 0)
            
            label = meta.get("class_label", "Local")
            
            xs.append(x)
            ys.append(y)
            zs.append(z)
            colors.append(COLOR_MAP.get(label, "#555555"))
            texts.append(create_hover_text(meta))
            
        trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                opacity=0.8,
                line=dict(width=0)
            ),
            text=texts,
            hoverinfo='text',
            name=model,
            visible=(i == 0) # Only first model visible initially
        )
        traces.append(trace)
        fig.add_trace(trace)
        
    # Build dropdown menu
    buttons = []
    for i, model in enumerate(MODELS):
        visibility = [j == i for j in range(len(MODELS))]
        buttons.append(dict(
            label=model,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"The Transformer Galaxy ({model})"}]
        ))
        
    fig.update_layout(
        title=f"The Transformer Galaxy ({MODELS[0]})",
        title_font=dict(color="#ffffff", size=24),
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top",
            font=dict(color="#ffffff")
        )],
        scene=dict(
            xaxis_title='Mean Attention Distance (Tokens)',
            yaxis_title='Entropy Collapse (-H)',
            zaxis_title='V/Q Ratio',
            xaxis=dict(backgroundcolor="black", gridcolor="#333", zerolinecolor="#555"),
            yaxis=dict(backgroundcolor="black", gridcolor="#333", zerolinecolor="#555"),
            zaxis=dict(backgroundcolor="black", gridcolor="#333", zerolinecolor="#555")
        ),
        paper_bgcolor='rgba(0,0,0,1)',
        plot_bgcolor='rgba(0,0,0,1)',
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    out_path = "website/public/microscope.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Generated microscope dashboard at {out_path}")

if __name__ == "__main__":
    build_dashboard()
