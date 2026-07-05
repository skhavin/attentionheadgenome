import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Seaborn for beautiful aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

os.makedirs('outputs/final_artifacts/visualizations', exist_ok=True)

# 1. Plot Routing Validation (Experiment A and B vs Baseline)
def plot_routing_validation():
    try:
        with open('outputs/routing/Qwen2.5-0.5B_validation_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Routing validation data not found.")
        return

    labels = ['Baseline', 'Exp A\n(Local\u2192Windowed)', 'Exp B\n(Sink\u2192BOS)']
    
    hellaswag = [
        data['baseline']['hellaswag'] * 100,
        data['exp_A']['hellaswag'] * 100,
        data['exp_B']['hellaswag'] * 100
    ]
    
    arc = [
        data['baseline']['arc'] * 100,
        data['exp_A']['arc'] * 100,
        data['exp_B']['arc'] * 100
    ]

    ppl = [
        data['baseline']['ppl'],
        data['exp_A']['ppl'],
        data['exp_B']['ppl']
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Accuracy
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, hellaswag, width, label='HellaSwag', color='#3498db', edgecolor='black', linewidth=1.2)
    ax1.bar(x + width/2, arc, width, label='ARC-Easy', color='#2ecc71', edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Out-of-Domain Generalization During Routing', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold')
    ax1.set_ylim(0, max(max(hellaswag), max(arc)) + 15)
    ax1.legend()

    # Add delta annotations for HellaSwag
    ax1.text(1 - width/2, hellaswag[1] + 1, f"{-1.0}%", ha='center', color='#e74c3c', fontweight='bold')
    ax1.text(2 - width/2, hellaswag[2] + 1, f"{-5.0}%", ha='center', color='#e74c3c', fontweight='bold')

    # Plot Perplexity
    ax2.bar(labels, ppl, color='#9b59b6', edgecolor='black', linewidth=1.2, width=0.5)
    ax2.set_ylabel('Perplexity (Lower is Better)', fontweight='bold')
    ax2.set_title('WikiText-103 Perplexity', fontweight='bold', pad=15)
    for i, p in enumerate(ppl):
        ax2.text(i, p + 0.5, f"{p:.1f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/final_artifacts/visualizations/routing_validation.png', dpi=300)
    plt.close()
    print("Saved routing validation visualization.")

# 2. Plot Taxonomy Distribution
def plot_taxonomy_distribution():
    models = ['gpt2-medium', 'Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Llama-3.2-1B']
    classes = ['Local', 'Sink', 'Induction', 'Retrieval', 'Unknown']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#95a5a6']
    
    distribution = {c: [] for c in classes}
    
    for model in models:
        try:
            with open(f'outputs/phase2_atlas/{model}_head_atlas.json', 'r') as f:
                atlas = json.load(f)
        except FileNotFoundError:
            for c in classes: distribution[c].append(0)
            continue
            
        counts = {c: 0 for c in classes}
        total = 0
        for head in atlas['heads'].values():
            counts[head.get('class_label', 'Unknown')] += 1
            total += 1
            
        for c in classes:
            distribution[c].append(counts[c] / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = np.zeros(len(models))
    for i, c in enumerate(classes):
        values = np.array(distribution[c])
        ax.bar(models, values, bottom=bottom, label=c, color=colors[i], edgecolor='white')
        bottom += values
        
    ax.set_ylabel('Percentage of Heads (%)', fontweight='bold')
    ax.set_title('Attention Head Taxonomy Distribution Across Models', fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('outputs/final_artifacts/visualizations/taxonomy_distribution.png', dpi=300)
    plt.close()
    print("Saved taxonomy distribution visualization.")

# 3. Plot UMAP Clusters (Emergent Discovery)
def plot_emergent_umap():
    try:
        import umap
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("UMAP or sklearn not installed. Skipping UMAP visualization.")
        return

    models = ['gpt2-medium', 'Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Llama-3.2-1B']
    
    rows = []
    labels = []
    
    for model in models:
        try:
            with open(f'outputs/phase2_atlas/{model}_head_atlas.json', 'r') as f:
                atlas = json.load(f)
            with open(f'outputs/routing/{model}_rich_features.json', 'r') as f:
                rich = json.load(f)
        except:
            continue
            
        for k, head in atlas['heads'].items():
            r = rich['heads'].get(k, {})
            geom = head.get('attention_geometry', {})
            sat = head.get('softmax_saturation', {})
            if not geom or not sat: continue
            
            row = [
                geom.get("bos_mass", 0.0),
                geom.get("local_mass", 0.0),
                geom.get("long_range_mass", 0.0),
                sat.get("mean_max_attn", 0.0),
                sat.get("mean_entropy", 0.0),
                r.get("activation_sparsity", 0.0) or 0.0,
                r.get("inter_layer_corr", 0.0) or 0.0,
            ]
            rows.append(row)
            labels.append(head.get('class_label', 'Unknown'))
            
    if not rows: return
    
    X = np.array(rows, dtype=float)
    X[np.isnan(X)] = 0
    X_scaled = StandardScaler().fit_transform(X)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    colors = {'Local': '#3498db', 'Sink': '#e74c3c', 'Induction': '#2ecc71', 'Retrieval': '#f1c40f', 'Unknown': '#95a5a6'}
    
    for label in ['Local', 'Sink', 'Induction', 'Retrieval']:
        mask = [l == label for l in labels]
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   c=colors[label], label=label, alpha=0.6, s=30, edgecolor='white', linewidth=0.5)
                   
    plt.title('UMAP Projection of Attention Heads (Emergent Features)', fontweight='bold', pad=15)
    plt.legend()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/final_artifacts/visualizations/umap_clusters.png', dpi=300)
    plt.close()
    print("Saved UMAP clusters visualization.")

if __name__ == '__main__':
    plot_routing_validation()
    plot_taxonomy_distribution()
    plot_emergent_umap()
