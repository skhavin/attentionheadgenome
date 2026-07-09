import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset

def compute_ppl(model, prompt_tensors, target_span=None):
    """
    Compute average perplexity over a list of prompt tensors.
    If target_span is provided as (start_idx, end_idx), PPL is only computed
    over the loss of those specific tokens (used for NIAH and Induction).
    """
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for tokens in prompt_tensors:
            tokens = tokens.to(model.device)
            out = model(tokens, labels=tokens)
            
            # out.logits: [batch, seq_len, vocab_size]
            # out.loss is the mean over the whole sequence.
            # We need to compute loss manually over the target span.
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            # flatten
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(tokens.size(0), -1) # [batch, seq_len - 1]
            
            if target_span is not None:
                start_idx, end_idx = target_span
                # The labels are shifted by 1, so index i in loss corresponds to predicting token i+1.
                # If we want to evaluate predicting tokens [start_idx, end_idx)
                # the loss indices are [start_idx - 1, end_idx - 1)
                span_loss = loss[:, start_idx-1 : end_idx-1]
                nll = span_loss.sum().item()
                n = end_idx - start_idx
            else:
                nll = loss.sum().item()
                n = tokens.shape[1] - 1
                
            total_nll += nll
            total_tokens += n

    return float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")


def load_wikitext_prompts(tokenizer, n=5, seq_len=256):
    """Load held-out evaluation prompts for Local/Sink heads."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = " ".join(ds["text"])
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompts = []
    stride = max(1, len(tokens) // n)
    for i in range(n):
        chunk = tokens[i * stride: i * stride + seq_len]
        if len(chunk) == seq_len:
            prompts.append(torch.tensor(chunk).unsqueeze(0))
    return prompts[:n]


def generate_niah_prompts(tokenizer, n=50, seq_len=256, shuffle_type="none"):
    """
    Generate NIAH prompts for Retrieval heads.
    Returns (prompts, target_span) where target_span is (start, end) of the needle answer.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = " ".join(ds["text"])
    base_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    prompts = []
    # Use a fixed query length so target_span is consistent from the end
    needle_val = " 8274193"
    needle_str = f" The special magic number for today is:{needle_val}."
    query_str = " What is the special magic number for today? The special magic number for today is:"
    
    needle_tokens = tokenizer.encode(needle_str, add_special_tokens=False)
    query_tokens = tokenizer.encode(query_str, add_special_tokens=False)
    ans_tokens = tokenizer.encode(needle_val, add_special_tokens=False)
    
    ans_len = len(ans_tokens)
    
    stride = max(1, len(base_tokens) // n)
    for i in range(n):
        chunk = base_tokens[i * stride: i * stride + seq_len - len(needle_tokens) - len(query_tokens) - ans_len]
        
        if shuffle_type == "content_shuffle":
            # Replace filler AND needle with random tokens, keeping query intact
            random_chunk = torch.randint(3, tokenizer.vocab_size, (len(chunk) + len(needle_tokens),)).tolist()
            prompt_list = random_chunk + query_tokens + ans_tokens
        else:
            if shuffle_type == "position_shuffle":
                # Shuffle the filler, but keep the needle intact (just placed randomly)
                chunk_tensor = torch.tensor(chunk)
                perm = torch.randperm(len(chunk))
                chunk = chunk_tensor[perm].tolist()
                
            depth_pct = (i % 3) * 0.4 + 0.1 # 0.1, 0.5, 0.9
            insert_idx = int(len(chunk) * depth_pct)
            prompt_list = chunk[:insert_idx] + needle_tokens + chunk[insert_idx:] + query_tokens + ans_tokens
            
        prompts.append(torch.tensor(prompt_list).unsqueeze(0))
        
    target_span = (len(prompt_list) - ans_len, len(prompt_list))
    return prompts, target_span


def generate_induction_prompts(tokenizer, n=50, seq_len=128, shuffle_type="none"):
    """
    Generate Induction prompts: [A][B]...[A]->[B]
    Returns (prompts, target_span) for the final [B].
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = " ".join(ds["text"])
    base_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    prompts = []
    stride = max(1, len(base_tokens) // n)
    target_span = (seq_len - 1, seq_len)
    
    for i in range(n):
        chunk = base_tokens[i * stride: i * stride + seq_len]
        if len(chunk) < seq_len:
            continue
            
        A = chunk[30]
        B = chunk[31]
        
        if shuffle_type == "content_shuffle":
            # Replace everything before the final A->B with random tokens (pattern is gone)
            random_chunk = torch.randint(3, tokenizer.vocab_size, (seq_len - 2,)).tolist()
            prompt_list = random_chunk + [A, B]
        elif shuffle_type == "position_shuffle":
            # Shuffle everything before the final A->B, breaking the first A->B pair
            prefix = torch.tensor(chunk[:-2])
            perm = torch.randperm(len(prefix))
            prompt_list = prefix[perm].tolist() + [A, B]
        else:
            # Normal: place A at the end to predict B
            chunk[-2] = A
            chunk[-1] = B
            prompt_list = chunk
            
        prompts.append(torch.tensor(prompt_list).unsqueeze(0))
        
    return prompts, target_span

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


def compute_head_delta_ppl(model, tokenizer, prompt_tensors, layer_idx, head_idx, 
                           intervention_fn, target="q", baseline_ppl=None,
                           architecture="unknown", prompt_id="unknown", label="unknown",
                           dry_run=False, condition_name="unknown"):
    """
    Evaluates the task-specific damage of an intervention on a head.
    Automatically routes to the correct task (NIAH/Induction/Wikitext) based on the label.
    """
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_heads = model.config.num_attention_heads
    
    # Extract shuffle_type if condition_name dictates it
    shuffle_type = "none"
    if "position_shuffle" in condition_name:
        shuffle_type = "position_shuffle"
    elif "content_shuffle" in condition_name:
        shuffle_type = "content_shuffle"
    
    # Task Routing
    if label == "retrieval":
        task_prompts, target_span = generate_niah_prompts(tokenizer, n=10 if dry_run else 50, shuffle_type=shuffle_type)
        eval_task = "niah_needle_span"
        scoring_span = f"[-{target_span[1]-target_span[0]}:]"
    elif label == "induction":
        task_prompts, target_span = generate_induction_prompts(tokenizer, n=10 if dry_run else 50, shuffle_type=shuffle_type)
        eval_task = "copy_repeat_span"
        scoring_span = "[-1:]"
    else:
        # Local, Sink, Unknown use standard WikiText
        task_prompts = prompt_tensors  # These should be passed in as standard WikiText or globally shuffled WikiText
        target_span = None
        eval_task = "wikitext_full"
        scoring_span = "all"
        
    # Recompute baseline if the task isn't the standard WikiText passed in
    if eval_task != "wikitext_full" or baseline_ppl is None:
        baseline_ppl = compute_ppl(model, task_prompts, target_span=target_span)

    attn_module = get_attn_module(model, layer_idx)
    if attn_module is None:
        raise ValueError(f"Could not find attention module for layer {layer_idx}")

    # For GQA, sibling heads must be isolated.
    if hasattr(model.config, "num_key_value_heads"):
        n_kv_heads = model.config.num_key_value_heads
    else:
        n_kv_heads = n_heads
    
    with HeadPatcher(
        model=model,
        layer_idx=layer_idx,
        head_idx=head_idx,
        intervention_fn=intervention_fn,
        target=target
    ) as patcher:
        intervened_ppl = compute_ppl(model, task_prompts, target_span=target_span)
        
    # Note: Output norm tracking is only strictly meaningful if we were recording it.
    # The patcher hook can compute it for OV zeroing if needed, but we focus on ΔPPL here.
    
    delta_ppl = intervened_ppl - baseline_ppl
    
    if dry_run:
        print(f"[DRY RUN] L{layer_idx}H{head_idx} ({label}) [{eval_task}]:")
        print(f"  Baseline PPL: {baseline_ppl:.4f}")
        print(f"  Intervened PPL: {intervened_ppl:.4f}")
        print(f"  Delta PPL: {delta_ppl:.4f}")
        
    return {
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "architecture": architecture,
        "intervention_type": condition_name,
        "prompt_id": prompt_id,
        "eval_task": eval_task,
        "scoring_span": scoring_span,
        "baseline_ppl": baseline_ppl,
        "intervened_ppl": intervened_ppl,
        "delta_ppl": delta_ppl,
        "canonical_label": label,
        "output_norm_baseline": None,
        "output_norm_intervened": None,
        "delta_output_norm": None
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

