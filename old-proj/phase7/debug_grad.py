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

dummy = torch.randint(0, 1000, (1, 64))
patcher.reset_loss()
_ = model(dummy)

# Check MSE
mse = patcher.accumulated_mse
print(f"MSE type: {type(mse)}")
print(f"MSE value: {mse}")
print(f"MSE is tensor: {isinstance(mse, torch.Tensor)}")
if isinstance(mse, torch.Tensor):
    print(f"MSE requires_grad: {mse.requires_grad}")
    print(f"MSE grad_fn: {mse.grad_fn}")

# Collect probs
all_probs = []
for r in patcher.routers.values():
    if r._last_probs_for_loss is not None:
        all_probs.append(r._last_probs_for_loss)

print(f"\nNum routers with probs: {len(all_probs)}")
print(f"probs[0].requires_grad: {all_probs[0].requires_grad}")
print(f"probs[0].grad_fn: {all_probs[0].grad_fn}")

# Entropy
probs_stacked = torch.stack(all_probs)
entropy = -(probs_stacked * (probs_stacked + 1e-8).log()).sum(dim=-1).mean()
print(f"\nEntropy: {entropy.item():.4f}")
print(f"Entropy requires_grad: {entropy.requires_grad}")
print(f"Entropy grad_fn: {entropy.grad_fn}")

# Total loss
loss = mse + 1.5 * entropy
print(f"\nTotal loss: {loss.item():.4f}")
print(f"Total loss requires_grad: {loss.requires_grad}")

loss.backward()

# Check gradients on first router
r0 = list(patcher.routers.values())[0]
print(f"\nfc1.weight.grad is None: {r0.fc1.weight.grad is None}")
print(f"fc2.weight.grad is None: {r0.fc2.weight.grad is None}")
print(f"fc2.bias.grad is None: {r0.fc2.bias.grad is None}")
if r0.fc2.bias.grad is not None:
    print(f"fc2.bias.grad norm: {r0.fc2.bias.grad.float().norm().item():.6f}")
    print(f"fc2.bias.grad[:8]: {r0.fc2.bias.grad[:8].float().tolist()}")
else:
    print("NO GRADIENTS ON ROUTER WEIGHTS - entropy loss not backpropping!")

# Check if mse has grad fn
print(f"\nAccumulated MSE grad_fn: {patcher.accumulated_mse.grad_fn if isinstance(patcher.accumulated_mse, torch.Tensor) else 'not a tensor'}")
