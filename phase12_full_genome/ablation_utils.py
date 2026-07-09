import sys
import torch
import torch.nn as nn
import numpy as np

def get_attn_module(model, layer_idx):
    """Find the attention module for a specific layer."""
    for name, module in model.named_modules():
        if f"layers.{layer_idx}" in name or f"h.{layer_idx}" in name:
            if any(t in type(module).__name__ for t in ["Attention", "attention"]):
                if hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                    return module
    raise ValueError(f"Could not find attention module for layer {layer_idx}")

class HeadPatcher:
    """
    Safely registers a forward hook to intervene on a specific head's Q or OV tensor.
    Guarantees isolation (GQA safe) and immediate removal.
    """
    def __init__(self, model, layer_idx, head_idx, intervention_fn, target="q"):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.intervention_fn = intervention_fn
        self.target = target # "q" or "ov"
        self.hook_handle = None
        
        self.attn_module = get_attn_module(self.model, self.layer_idx)
        self.n_heads = self.model.config.num_attention_heads

        # Identify which sub-module to hook based on architecture and target
        self.is_gpt2 = hasattr(self.attn_module, "c_attn")
        self.target_module = None
        
        if self.target == "q":
            self.target_module = self.attn_module.c_attn if self.is_gpt2 else self.attn_module.q_proj
        elif self.target == "ov":
            self.target_module = self.attn_module.c_proj if self.is_gpt2 else self.attn_module.o_proj
        else:
            raise ValueError(f"Unknown target {self.target}")

    def __enter__(self):
        # We need distinct logic for Q (post-hook) and OV (pre-hook)
        if self.target == "q":
            self.hook_handle = self.target_module.register_forward_hook(self._q_hook)
        elif self.target == "ov":
            self.hook_handle = self.target_module.register_forward_pre_hook(self._ov_hook)
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _q_hook(self, module, inputs, output):
        batch, seq, out_dim = output.shape
        head_dim = (out_dim // 3) // self.n_heads if self.is_gpt2 else out_dim // self.n_heads
        
        if self.is_gpt2:
            # Output is [batch, seq, 3 * hidden]. Q is the first third.
            hidden = out_dim // 3
            q_chunk = output[:, :, :hidden]
            k_chunk = output[:, :, hidden:2*hidden]
            v_chunk = output[:, :, 2*hidden:]
            
            q_reshaped = q_chunk.view(batch, seq, self.n_heads, head_dim)
            q_slice = q_reshaped[:, :, self.head_idx, :]
            
            modified_q_slice = self.intervention_fn(q_slice.clone())
            assert modified_q_slice.shape == q_slice.shape, "Intervention function changed tensor shape!"
            
            q_reshaped[:, :, self.head_idx, :] = modified_q_slice
            q_restored = q_reshaped.view(batch, seq, hidden)
            return torch.cat([q_restored, k_chunk, v_chunk], dim=-1)
            
        else:
            # Output is [batch, seq, n_heads * head_dim]
            q_reshaped = output.view(batch, seq, self.n_heads, head_dim)
            q_slice = q_reshaped[:, :, self.head_idx, :]
            
            modified_q_slice = self.intervention_fn(q_slice.clone())
            assert modified_q_slice.shape == q_slice.shape, "Intervention function changed tensor shape!"
            
            q_reshaped[:, :, self.head_idx, :] = modified_q_slice
            return q_reshaped.view(batch, seq, out_dim)

    def _ov_hook(self, module, inputs):
        # inputs is a tuple, inputs[0] is the concatenated head outputs [batch, seq, hidden]
        inp = inputs[0]
        batch, seq, hidden = inp.shape
        head_dim = hidden // self.n_heads
        
        inp_reshaped = inp.view(batch, seq, self.n_heads, head_dim)
        ov_slice = inp_reshaped[:, :, self.head_idx, :]
        
        # Calculate output norm for diagnostic BEFORE intervention
        # We attach it to the module temporarily so we can retrieve it
        norm = torch.linalg.norm(ov_slice, dim=-1).mean().item()
        module._current_output_norm = norm
        
        modified_ov_slice = self.intervention_fn(ov_slice.clone())
        assert modified_ov_slice.shape == ov_slice.shape, "Intervention function changed tensor shape!"
        
        inp_reshaped[:, :, self.head_idx, :] = modified_ov_slice
        
        # Calculate intervened norm
        int_norm = torch.linalg.norm(modified_ov_slice, dim=-1).mean().item()
        module._current_intervened_norm = int_norm
        
        return (inp_reshaped.view(batch, seq, hidden),) + inputs[1:]


def compute_head_delta_ppl(model, tokenizer, prompt_tensors, layer_idx, head_idx, intervention_fn, target="q", baseline_ppl=None, architecture="unknown", prompt_id="unknown", label="unknown", dry_run=False, condition_name="unknown"):
    """
    Unified metric-collection function.
    """
    assert len(model._forward_hooks) == 0, "Model has leaked hooks before starting baseline!"
    assert len(model._forward_pre_hooks) == 0, "Model has leaked pre-hooks before starting baseline!"
    
    device = model.device
    
    # 1. Compute baseline if not provided
    if baseline_ppl is None:
        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for p in prompt_tensors:
                p = p.to(device)
                out = model(p, labels=p)
                nll = out.loss.item()
                n = p.shape[1] - 1
                total_nll += nll * n
                total_tokens += n
        baseline_ppl = float(np.exp(total_nll / total_tokens))
        
    # 2. Intervene and compute PPL
    intervened_nll = 0.0
    total_tokens = 0
    norm_baseline = 0.0
    norm_intervened = 0.0
    
    with HeadPatcher(model, layer_idx, head_idx, intervention_fn, target=target) as patcher:
        with torch.no_grad():
            for p in prompt_tensors:
                p = p.to(device)
                out = model(p, labels=p)
                nll = out.loss.item()
                n = p.shape[1] - 1
                intervened_nll += nll * n
                total_tokens += n
                
                if target == "ov":
                    norm_baseline += getattr(patcher.target_module, "_current_output_norm", 0.0)
                    norm_intervened += getattr(patcher.target_module, "_current_intervened_norm", 0.0)

    intervened_ppl = float(np.exp(intervened_nll / total_tokens))
    delta_ppl = intervened_ppl - baseline_ppl
    
    # Average the norms over the prompts
    num_prompts = len(prompt_tensors)
    norm_baseline /= num_prompts
    norm_intervened /= num_prompts

    if dry_run:
        print(f"[DRY RUN] L{layer_idx}H{head_idx} ({label}):")
        print(f"  Baseline PPL: {baseline_ppl:.4f}")
        print(f"  Intervened PPL: {intervened_ppl:.4f}")
        print(f"  Delta PPL: {delta_ppl:.4f}")
        if target == "ov":
            print(f"  Output Norm Baseline: {norm_baseline:.4f}")
            print(f"  Output Norm Intervened: {norm_intervened:.4f}")

    return {
        "layer_idx": int(layer_idx),
        "head_idx": int(head_idx),
        "architecture": architecture,
        "intervention_type": condition_name,
        "prompt_id": prompt_id,
        "baseline_ppl": float(baseline_ppl),
        "intervened_ppl": float(intervened_ppl),
        "delta_ppl": float(delta_ppl),
        "canonical_label": label,
        "output_norm_baseline": float(norm_baseline) if target == "ov" else None,
        "output_norm_intervened": float(norm_intervened) if target == "ov" else None,
        "delta_output_norm": float(norm_intervened - norm_baseline) if target == "ov" else None
    }


def q_permutation_fn(q_slice):
    """
    Permutes the sequence dimension of the Q tensor.
    Input shape: [batch, seq, head_dim]
    """
    batch, seq, head_dim = q_slice.shape
    perm = torch.randperm(seq, device=q_slice.device)
    return q_slice[:, perm, :]


def ov_zero_fn(ov_slice):
    """
    Zeros out the OV output tensor.
    Input shape: [batch, seq, head_dim]
    """
    return torch.zeros_like(ov_slice)


def test_gqa_isolation(model, tokenizer):
    """
    Unit test to verify that patching a single Q-head in a GQA architecture
    strictly isolates that head without touching sibling heads in the same KV group.
    """
    # Find the Q-projection layer
    attn_module = get_attn_module(model, 0)
    
    is_gpt2 = hasattr(attn_module, "c_attn")
    if is_gpt2:
        return True # MHA doesn't have GQA sharing issues
        
    n_heads = model.config.num_attention_heads
    target_head = 0
    sibling_head = 1 # In Llama and Qwen, head 0 and 1 usually share a KV head or are simply adjacent
    
    # Get pre-intervention baseline
    device = model.device
    inp = torch.randint(0, 1000, (1, 10), device=device)
    
    baseline_q = None
    def capture_hook(module, inputs, output):
        nonlocal baseline_q
        baseline_q = output.clone()
    
    h = attn_module.q_proj.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(inp)
    h.remove()
    
    # Now run with intervention
    intervened_q = None
    def capture_intervened(module, inputs, output):
        nonlocal intervened_q
        intervened_q = output.clone()
        
    with HeadPatcher(model, 0, target_head, q_permutation_fn, target="q") as patcher:
        h2 = patcher.target_module.register_forward_hook(capture_intervened)
        with torch.no_grad():
            model(inp)
        h2.remove()
        
    # Check bit-identity
    batch, seq, out_dim = baseline_q.shape
    head_dim = out_dim // n_heads
    
    base_reshaped = baseline_q.view(batch, seq, n_heads, head_dim)
    int_reshaped = intervened_q.view(batch, seq, n_heads, head_dim)
    
    # Target head should be different
    target_base = base_reshaped[:, :, target_head, :]
    target_int = int_reshaped[:, :, target_head, :]
    assert not torch.equal(target_base, target_int), "Target head was not modified!"
    
    # Sibling head should be BIT-IDENTICAL
    sibling_base = base_reshaped[:, :, sibling_head, :]
    sibling_int = int_reshaped[:, :, sibling_head, :]
    assert torch.equal(sibling_base, sibling_int), "GQA Isolation Failure: Sibling head was modified!"
    
    print(f"[PASS] GQA Isolation Test for {model.config._name_or_path}: Sibling heads perfectly preserved.")
    return True

