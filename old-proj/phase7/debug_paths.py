"""
Check per-path MSE quality: how well does each path approximate full attention?
This tells us whether the cheap paths are even learnable targets.
"""
import sys, torch
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM
from phase7.moe.paths import sink_path, local_path, recurrence_path
from phase7.moe.moe_patcher import MoEPatcher
import json

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B', device_map='cpu',
    torch_dtype=torch.bfloat16, attn_implementation='eager'
)
for p in model.parameters():
    p.requires_grad = False

# Load real data
with open('data/stage2_qwen_full.jsonl') as f:
    data = [json.loads(l) for l in f.readlines()[:4]]

# Hook to capture V and full attention output per layer
captured = {}
hooks = []

def make_hooks(model):
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if 'Attention' in cls_name and hasattr(module, 'q_proj'):
            parts = name.split('.')
            layer_idx = next((int(p) for p in parts if p.isdigit()), 0)
            
            def v_hook(m, inp, out, lidx=layer_idx):
                B, N, d = out.shape
                nkv = model.config.num_key_value_heads
                nh = model.config.num_attention_heads
                hidden_size = model.config.hidden_size
                dh = hidden_size // nh
                v = out.view(B, N, nkv, dh)
                groups = nh // nkv
                if groups > 1:
                    v = v.repeat_interleave(groups, dim=2)
                captured[f'v_{lidx}'] = v.transpose(1, 2).detach()  # [B, H, N, d]

            def out_hook(m, args, lidx=layer_idx):
                captured[f'full_{lidx}'] = args[0].detach()  # [B, N, H*d]

            hooks.append(module.v_proj.register_forward_hook(v_hook))
            hooks.append(module.o_proj.register_forward_pre_hook(out_hook))

make_hooks(model)

# Run a batch
batch = torch.tensor([data[0]['tokens'][:256]])
with torch.no_grad():
    model(batch)

# Per-layer per-path MSE
nh = model.config.num_attention_heads
dh = model.config.hidden_size // nh

print(f"{'Layer':>6} {'sink_mse':>10} {'local_mse':>10} {'rec_mse':>10}")
print("-" * 42)

for lidx in range(24):
    if f'v_{lidx}' not in captured:
        continue
    V = captured[f'v_{lidx}']          # [B, H, N, d]
    full_out = captured[f'full_{lidx}'] # [B, N, H*d]
    B, H, N, d = V.shape
    
    # Reconstruct what each path produces after o_proj would be applied
    # Compare V-space outputs (before o_proj) normalized by full_out scale
    sink_out = sink_path(V, 4)
    local_out = local_path(V, 64)
    rec_out = recurrence_path(V, 0.9)
    full_v = full_out.view(B, N, H, d).transpose(1, 2)  # [B,H,N,d]
    
    sink_mse = torch.nn.functional.mse_loss(sink_out.float(), full_v.float()).item()
    local_mse = torch.nn.functional.mse_loss(local_out.float(), full_v.float()).item()
    rec_mse = torch.nn.functional.mse_loss(rec_out.float(), full_v.float()).item()
    
    print(f"  l{lidx:>2}   {sink_mse:>10.4f} {local_mse:>10.4f} {rec_mse:>10.4f}")

for h in hooks:
    h.remove()
