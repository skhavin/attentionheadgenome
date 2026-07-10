import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=22)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    prompt = "The capital of France is Paris. The capital of Germany is"
    print(f"\nPrompt: '{prompt}'")
    
    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    layer_idx = args.layer
    layer = model.model.layers[layer_idx]
    
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    head_outputs = {}
    
    # We hook into the self_attn o_proj to capture the input (which is the concatenated head outputs)
    def o_proj_hook(module, inp, out):
        # inp[0] is shape (batch, seq_len, hidden_size)
        # It contains the concatenated outputs of all V heads before W_O is applied.
        v_concat = inp[0][0, -1, :]  # shape: (hidden_size,)
        
        for h in range(n_heads):
            # Isolate the V vector for head h, zero out the rest
            v_isolated = torch.zeros_like(v_concat)
            v_isolated[h * head_dim : (h+1) * head_dim] = v_concat[h * head_dim : (h+1) * head_dim]
            
            # Pass through W_O to get the final update vector for this head
            head_update = F.linear(v_isolated, module.weight, module.bias)
            head_outputs[h] = head_update
            
    hook_handle = layer.self_attn.o_proj.register_forward_hook(o_proj_hook)
    
    with torch.no_grad():
        model(**tokens)
        
    hook_handle.remove()
    
    unembed = model.lm_head
    final_norm = model.model.norm
    
    print(f"\n======================================")
    print(f"Direct Logit Attribution (Layer {layer_idx})")
    print(f"======================================")
    
    # Check what each head is writing to the output vocabulary
    for h in range(n_heads):
        update = head_outputs[h] # (hidden_size,)
        
        # In strict DLA, we apply the final LayerNorm scaling factor from the actual residual stream.
        # For simplicity, we just project the update directly through the unembedding.
        # (Technically, the residual stream is LayerNormed before lm_head, but projecting the pre-norm update 
        # still reveals the direction the head is pushing the logits).
        
        logits = unembed(update.to(model.dtype))
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_indices = torch.topk(logits, 5)
        
        print(f"Head {h}:")
        for val, idx in zip(top_vals, top_indices):
            pred_word = tokenizer.decode([idx.item()])
            print(f"  -> '{pred_word}' (Logit: {val.item():.2f})")
        print()

if __name__ == "__main__":
    main()
