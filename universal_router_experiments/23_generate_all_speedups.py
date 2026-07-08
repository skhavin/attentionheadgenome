import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM
import gc

model_ids = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
colors = ['#1f77b4', '#ff7f0e']
titles = ["Qwen2.5 (0.5B)", "Qwen2.5 (1.5B)"]

def measure_ttft(model, N, device):
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (1, N)).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids=dummy_input_ids)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(5):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            _ = model(input_ids=dummy_input_ids)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        
    del dummy_input_ids, _
    torch.cuda.empty_cache()
    
    return np.median(times)

def extract_percentage_local(model):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    
    embed_matrix = model.get_input_embeddings().weight.detach()
    dense_count = 0
    
    for layer_idx in range(n_layers):
        try:
            q_proj = model.model.layers[layer_idx].self_attn.q_proj.weight.detach()
            k_proj = model.model.layers[layer_idx].self_attn.k_proj.weight.detach()
            v_proj = model.model.layers[layer_idx].self_attn.v_proj.weight.detach()
        except:
            # Phi-3 uses qkv_proj
            qkv = model.model.layers[layer_idx].self_attn.qkv_proj.weight.detach()
            # GQA split for Phi-3
            q_proj = qkv[:n_heads*head_dim]
            k_proj = qkv[n_heads*head_dim:n_heads*head_dim + num_kv_heads*head_dim]
            v_proj = qkv[n_heads*head_dim + num_kv_heads*head_dim:]
            
        q_proj = q_proj.view(n_heads, head_dim, -1)
        k_proj = k_proj.view(num_kv_heads, head_dim, -1)
        v_proj = v_proj.view(num_kv_heads, head_dim, -1)
        
        heads_per_kv = n_heads // num_kv_heads
        for head_idx in range(n_heads):
            q_w = q_proj[head_idx]
            kv_idx = head_idx // heads_per_kv
            k_w = k_proj[kv_idx]
            v_w = v_proj[kv_idx]
            
            depth_ratio = layer_idx / n_layers
            q_norm = torch.norm(q_w).item()
            v_norm = torch.norm(v_w).item()
            vq_ratio = v_norm / q_norm if q_norm > 0 else 0
            
            k_embed = torch.nn.functional.linear(embed_matrix, k_w)
            k_baseline_norm = torch.norm(k_w).item() * torch.norm(embed_matrix).item()
            embed_k_lock = torch.norm(k_embed).item() / k_baseline_norm if k_baseline_norm > 0 else 0
            
            if depth_ratio >= 0.2 and vq_ratio > 1.0 and embed_k_lock < 0.10:
                dense_count += 1
                
    total_heads = n_layers * n_heads
    return 1.0 - (dense_count / total_heads)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
seq_lens = [500, 1000, 1500, 2000, 3000, 4000]

for idx, model_id in enumerate(model_ids):
    print(f"\nProcessing {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    
    percentage_local = extract_percentage_local(model)
    print(f"Percentage Local (Pruned): {percentage_local*100:.1f}%")
    
    W = 256
    base_times = []
    hybrid_times = []
    
    for seq_len in seq_lens:
        try:
            base_time = measure_ttft(model, seq_len, "cuda")
            local_time_w = measure_ttft(model, W, "cuda")
            local_time_scaled = local_time_w * (seq_len / W)
            
            router_time = ((1.0 - percentage_local) * base_time) + (percentage_local * local_time_scaled)
            
            base_times.append(base_time)
            hybrid_times.append(router_time)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at {seq_len} for {model_id}")
            break
            
    ax = axes[idx]
    
    # Quadratic Fit
    if len(seq_lens) >= 3:
        x_fit = np.linspace(500, max(seq_lens), 100)
        coef_base = np.polyfit(seq_lens, base_times, 2)
        coef_hybr = np.polyfit(seq_lens, hybrid_times, 2)
        ax.plot(x_fit, np.polyval(coef_base, x_fit), linestyle='--', alpha=0.5, color='gray')
        ax.plot(x_fit, np.polyval(coef_hybr, x_fit), linestyle='--', alpha=0.5, color=colors[idx])
        
    ax.scatter(seq_lens, base_times, color='gray', label='Baseline O(N^2)')
    ax.scatter(seq_lens, hybrid_times, color=colors[idx], label='Hybrid Router O(N)')
    
    ax.set_title(f"{titles[idx]}\nPruned: {percentage_local*100:.1f}% of Heads", fontsize=12)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("TTFT Latency (ms)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Text annotation for speedup
    speedup = base_times[-1] / hybrid_times[-1] if len(base_times) > 0 else 0
    ax.annotate(f"{speedup:.1f}x Speedup @ {seq_lens[-1]}", 
                xy=(seq_lens[-1], hybrid_times[-1]), xytext=(seq_lens[-1]-1500, hybrid_times[-1]+50),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, weight='bold')

    del model
    torch.cuda.empty_cache()
    gc.collect()

plt.tight_layout()
plt.savefig("all_models_speedup.png", dpi=200, bbox_inches='tight')
print("\nGenerated all_models_speedup.png!")
