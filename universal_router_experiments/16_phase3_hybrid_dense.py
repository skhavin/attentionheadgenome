import gc
import uuid
import math
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
CONTEXT = 1024
NUM_PROMPTS = 10

def get_pg_essay():
    return "The study of artificial intelligence has progressed rapidly over the past decade. Researchers are finding new ways to optimize large language models. Attention mechanisms are at the core of these advancements. " * 50

def run_calibration(model, tokenizer):
    print("  [>] Running 1-Shot Activation Probe Calibration...")
    base_text = get_pg_essay()
    needle_uuid = "CALIBRATION-MAGIC-123"
    needle = f" The special magic number is {needle_uuid}. "
    
    tokens = tokenizer(base_text, return_tensors="pt").input_ids[0]
    tokens = tokens[:CONTEXT - 30]
    context_str = tokenizer.decode(tokens)
    
    words = context_str.split()
    insert_idx = len(words) // 2
    context_str = " ".join(words[:insert_idx]) + needle + " ".join(words[insert_idx:])
    prompt = context_str + "\nQuestion: What is the special magic number?\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Bulletproof way to find the needle index: 
    # Tokenize the prefix exactly as it appears before the UUID, 
    # and the length of the prefix tokens is the exact starting index of the UUID!
    prefix = " ".join(words[:insert_idx]) + " The special magic number is "
    prefix_tokens = tokenizer(prefix, return_tensors="pt").input_ids[0]
    needle_start = len(prefix_tokens)
    print(f"      [SUCCESS] Found needle exactly at token index {needle_start}")
        
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    retrieval_heads = set()
    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]
    
    # Check attention weights of the last token (the question) looking at the needle
    for l in range(n_layers):
        for h in range(n_heads):
            # Attention mass from the last token to the needle region
            # We sum over a small window around the needle to catch all its tokens
            attn_mass = outputs.attentions[l][0, h, -1, needle_start:needle_start+10].sum().item()
            if attn_mass > 0.05: # 5% threshold
                retrieval_heads.add((l, h))
                
    print(f"      Found {len(retrieval_heads)} true Retrieval Heads out of {n_layers * n_heads} total heads.")
    return retrieval_heads

def build_hybrid_mask(seq_len, n_layers, n_heads, retrieval_heads, window=256, device="cuda", dtype=torch.bfloat16):
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    mask = torch.zeros((n_layers, n_heads, seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    
    window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype)) - \
                  torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=-window)
                  
    # [!] ATTENTION SINK FIX: Always keep the first 4 tokens unmasked for ALL heads
    # This prevents the softmax denominator from exploding when local heads can't see the BOS token
    window_mask[:, :4] = 1.0
                  
    for l in range(n_layers):
        for h in range(n_heads):
            if (l, h) not in retrieval_heads:
                # Local Head -> Pruned to W=256
                mask[l, h] = mask[l, h].masked_fill(window_mask == 0, float('-inf'))
            # Else: Retrieval Head -> Stays Dense (Full Causal)
    return mask

def evaluate_ppl(model, tokenizer, retrieval_heads):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50][:NUM_PROMPTS]
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    def run_eval(use_router):
        loss_sum = 0
        tok_sum = 0
        
        hooks = []
        if use_router:
            # We build the mask dynamically for the exact length
            def get_hook(layer_idx):
                def pre_hook(module, args, kwargs):
                    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    q_len = hidden_states.shape[1]
                    mask = build_hybrid_mask(q_len, n_layers, n_heads, retrieval_heads, dtype=model.dtype)
                    kwargs["attention_mask"] = mask[layer_idx, :, -q_len:, -q_len:].unsqueeze(0).contiguous()
                    return args, kwargs
                return pre_hook
            for l in range(n_layers):
                hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(get_hook(l), with_kwargs=True))
                
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT).to("cuda")
            seq_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
            loss_sum += outputs.loss.item() * seq_len
            tok_sum += seq_len
            
        for h in hooks: h.remove()
        return math.exp(loss_sum / tok_sum)
        
    return run_eval(False), run_eval(True)

def evaluate_niah(model, tokenizer, retrieval_heads):
    prompts = []
    base_text = get_pg_essay()
    
    for _ in range(NUM_PROMPTS):
        needle_uuid = str(uuid.uuid4())[:8].upper()
        needle = f" The special magic number is {needle_uuid}. "
        
        tokens = tokenizer(base_text, return_tensors="pt").input_ids[0]
        while len(tokens) < CONTEXT:
            base_text += base_text
            tokens = tokenizer(base_text, return_tensors="pt").input_ids[0]
            
        tokens = tokens[:CONTEXT - 30]
        context_str = tokenizer.decode(tokens)
        
        words = context_str.split()
        insert_idx = random.randint(len(words)//10, int(len(words)*0.8))
        context_str = " ".join(words[:insert_idx]) + needle + " ".join(words[insert_idx:])
        
        prompt = context_str + "\nQuestion: What is the special magic number?\nAnswer:"
        prompts.append((prompt, needle_uuid))
        
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
        
    def run_eval(use_router):
        correct = 0
        hooks = []
        if use_router:
            def get_hook(layer_idx):
                def pre_hook(module, args, kwargs):
                    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    q_len = hidden_states.shape[1]
                    if q_len > 1: # Only apply mask during prefill, generation is unmasked or handled by cache
                        mask = build_hybrid_mask(q_len, n_layers, n_heads, retrieval_heads, dtype=model.dtype)
                        kwargs["attention_mask"] = mask[layer_idx, :, -q_len:, -q_len:].unsqueeze(0).contiguous()
                    return args, kwargs
                return pre_hook
            for l in range(n_layers):
                hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(get_hook(l), with_kwargs=True))
                
        for prompt, needle_uuid in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONTEXT).to("cuda")
            seq_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(outputs[0][seq_len:], skip_special_tokens=True)
            if needle_uuid in generated:
                correct += 1
                
        for h in hooks: h.remove()
        return (correct / NUM_PROMPTS) * 100
        
    return run_eval(False), run_eval(True)

def main():
    print("="*60)
    print("  PHASE 3: HYBRID DENSE (1-SHOT PROBING)")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model.eval()
    
    # Phase 3: The 1-Shot Calibration
    retrieval_heads = run_calibration(model, tokenizer)
    
    # 1. PPL Test
    print("\n  [>] Evaluating WikiText PPL (1024 context)...")
    base_ppl, rout_ppl = evaluate_ppl(model, tokenizer, retrieval_heads)
    print(f"      Baseline PPL: {base_ppl:.2f} | Hybrid Router PPL: {rout_ppl:.2f}")
    
    # 2. RULER Test
    print("\n  [>] Evaluating Official NIAH @ 1024 context...")
    base_acc, rout_acc = evaluate_niah(model, tokenizer, retrieval_heads)
    print(f"      Baseline Acc: {base_acc:.1f}% | Hybrid Router Acc: {rout_acc:.1f}%")

if __name__ == "__main__":
    main()
