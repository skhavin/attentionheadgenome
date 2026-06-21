import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase7.audit_heads import detect_arch, iter_attn_layers

model = AutoModelForCausalLM.from_pretrained('gpt2-medium', torch_dtype=torch.float16, device_map='auto')
arch = detect_arch(model)
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]

layer_idx = 0
head_idx = 0
device = model.device

V_captured = {}
def _hook(module, inp, output):
    print("Hook called!")
    print(f"inp length: {len(inp)}")
    hidden_states = inp[0]
    print(f"hidden_states shape: {hidden_states.shape}")
    B, N, d_model = hidden_states.shape
    try:
        if hasattr(module, "c_attn"):
            print("Has c_attn")
            qkv = module.c_attn(hidden_states)
            print("c_attn succeeded")
            num_heads = getattr(module, "num_heads", None)
            if num_heads is None:
                num_heads = module.config.n_head # WRONG, it's model.config not module.config if it doesn't have it
            d_head = d_model // num_heads
            _, _, v = qkv.split(d_model, dim=2)
            v_heads = v.view(B, N, num_heads, d_head)
            V_captured["V"] = v_heads[:, :, head_idx, :].detach().cpu()
    except Exception as e:
        print(f"Hook exception: {e}")

layers = list(iter_attn_layers(model, arch))
_, attn_module = layers[layer_idx]
handle = attn_module.register_forward_hook(_hook)
with torch.no_grad():
    model(input_ids.to(device), use_cache=False)
handle.remove()

print(f"V is {V_captured.get('V')}")
