"""
Investigate why entropy gradient is tiny (0.002) and whether MSE grad flows.
"""
import sys, torch
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM
from phase7.moe.moe_patcher import MoEPatcher

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B', device_map='cpu',
    torch_dtype=torch.bfloat16, attn_implementation='eager'
)
for p in model.parameters():
    p.requires_grad = False

patcher = MoEPatcher(model, hard_routing=False)
for router in patcher.routers.values():
    router.to(model.dtype)
    for p in router.parameters():
        p.requires_grad = True

dummy = torch.randint(0, 1000, (1, 128))
patcher.reset_loss()
patcher.training = True
_ = model(dummy)
patcher.training = False

mse = patcher.accumulated_mse
print(f"MSE type: {type(mse)}")
print(f"MSE is tensor: {isinstance(mse, torch.Tensor)}")
if isinstance(mse, torch.Tensor):
    print(f"MSE value: {mse.item():.6f}")
    print(f"MSE requires_grad: {mse.requires_grad}")
    print(f"MSE grad_fn: {mse.grad_fn}")

# Collect probs
all_probs = []
for r in patcher.routers.values():
    if r._last_probs_for_loss is not None:
        all_probs.append(r._last_probs_for_loss)

probs_stacked = torch.stack(all_probs)
entropy = -(probs_stacked * (probs_stacked + 1e-8).log()).sum(dim=-1).mean()

# Test entropy-only loss
entropy.backward(retain_graph=True)
r0 = list(patcher.routers.values())[0]
grad_norm_entropy_only = r0.fc2.bias.grad.float().norm().item() if r0.fc2.bias.grad is not None else 0
print(f"\nEntropy-only grad norm on fc2.bias: {grad_norm_entropy_only:.6f}")

# Reset and test MSE-only loss
for r in patcher.routers.values():
    for p in r.parameters():
        if p.grad is not None:
            p.grad.zero_()

if isinstance(mse, torch.Tensor) and mse.requires_grad:
    mse.backward(retain_graph=True)
    grad_norm_mse_only = r0.fc2.bias.grad.float().norm().item() if r0.fc2.bias.grad is not None else 0
    print(f"MSE-only grad norm on fc2.bias: {grad_norm_mse_only:.6f}")
else:
    print("MSE has no grad — checking if patcher.training was set during forward pass")
    print("MSE won't backprop without patcher.training=True during forward")

# Print current bias values to see how far from uniform
print(f"\nCurrent fc2.bias[:8]: {r0.fc2.bias[:8].float().tolist()}")
print(f"Profile A init should be [-0.5,-0.5,-0.5,1.0] repeating")
print(f"After 50 steps with grad_norm=0.002 and lr=1e-4, weight change ≈ {50 * 0.002 * 1e-4:.6f}")
print(f"Profile A full-attention bias = 1.0, needs to drop to ~0 to start routing cheap paths")
print(f"Steps needed at current rate ≈ {1.0 / (0.002 * 1e-4):.0f} steps")
