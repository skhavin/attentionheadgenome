
import torch
from transformers import AutoConfig
from phase7.moe.moe_patcher import MoEPatcher
from phase7.moe.router import LayerRouter

model_name = "Qwen/Qwen2.5-0.5B"
config = AutoConfig.from_pretrained(model_name)
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads

print(f"Model: {model_name}")
print(f"Num layers: {num_layers}")
print(f"Num heads: {num_heads}")

# Count params for one layer router
router = LayerRouter(num_heads)
layer_params = sum(p.numel() for p in router.parameters())
print(f"Params per layer router: {layer_params}")

total_params = layer_params * num_layers
print(f"Total router params: {total_params}")
