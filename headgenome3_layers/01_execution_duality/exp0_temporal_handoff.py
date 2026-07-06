import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from headgenome2_circuits.utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase3_execution"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATASET_PATH = "headgenome2_circuits/datasets/arithmetic.json"

class PhaseActivationHook:
    """
    Hooks the input to o_proj to calculate the L2 norm of each head's contribution
    to the residual stream: || W_O^h * x^h ||_2
    Separates the calculation into Prefill (seq_len > 1) and Decode (seq_len == 1).
    """
    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []
        # Store average norms: dict[(layer, head)] = {"prefill": list(), "decode": list()}
        self.norms = {}
        
    def _create_hook(self, layer_idx):
        def hook(module, input):
            x = input[0] # Shape: (batch, seq_len, hidden_size)
            batch_size, seq_len, _ = x.shape
            
            phase = "prefill" if seq_len > 1 else "decode"
            
            # W_O shape: (hidden_size, hidden_size)
            W_O = module.weight # (out_features, in_features)
            
            for h in range(self.num_heads):
                start = h * self.head_dim
                end = start + self.head_dim
                
                # Input for this head: (batch, seq_len, head_dim)
                x_h = x[:, :, start:end]
                # Weights for this head: (hidden_size, head_dim)
                W_O_h = W_O[:, start:end]
                
                # Output contribution of this head: (batch, seq_len, hidden_size)
                head_out = F.linear(x_h, W_O_h)
                
                # Calculate L2 norm along hidden_size dimension
                norm = torch.linalg.norm(head_out, dim=-1) # (batch, seq_len)
                
                # Average over sequence length and batch
                avg_norm = norm.mean().item()
                
                key = (layer_idx, h)
                if key not in self.norms:
                    self.norms[key] = {"prefill": [], "decode": []}
                self.norms[key][phase].append(avg_norm)
                
            return (x,)
        return hook

    def register(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.o_proj.register_forward_pre_hook(self._create_hook(layer_idx))
            self.hooks.append(handle)
            
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def find_operand_indices(input_ids, tokenizer, x, y):
    tokens = [tokenizer.decode([tok]) for tok in input_ids[0]]
    x_str = str(x)
    y_str = str(y)
    
    x_idx, y_idx = -1, -1
    for i, tok in enumerate(tokens):
        if x_str in tok and x_idx == -1:
            x_idx = i
        elif y_str in tok and y_idx == -1:
            y_idx = i
            
    return x_idx, y_idx

def run_experiment_0(model_key="qwen-0.5b", num_samples=50):
    print(f"Loading {model_key} for Exp 0: Temporal Handoff...")
    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=True)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)[:num_samples]
        
    # Step 1: Identify Retrieval Heads (Prefill) and Late Induction Heads (Decode)
    retrieval_masses = torch.zeros((num_layers, num_heads), device=model.device)
    induction_masses = torch.zeros((num_layers, num_heads), device=model.device)
    
    valid_count = 0
    for item in tqdm(dataset, desc="Profiling Heads"):
        prompt = item["prompt"] # "Question: What is 4 plus 3? Answer: The sum is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        x, y = item["x"], item["y"]
        x_idx, y_idx = find_operand_indices(input_ids, tokenizer, x, y)
        
        # Find ' plus' token
        tokens = [tokenizer.decode([tok]) for tok in input_ids[0]]
        plus_idx = -1
        for i, tok in enumerate(tokens):
            if "plus" in tok.lower():
                plus_idx = i
                break
                
        if x_idx == -1 or y_idx == -1 or plus_idx == -1:
            continue
            
        valid_count += 1
        with torch.no_grad():
            outputs = model(**inputs)
            
        attentions = outputs.attentions
        last_tok_idx = input_ids.shape[1] - 1
        
        for l in range(num_layers):
            attn = attentions[l][0]
            # Retrieval: Attention from 'plus' to 'X' and 'Y'
            r_mass = attn[:, plus_idx, x_idx] + attn[:, plus_idx, y_idx]
            retrieval_masses[l] += r_mass
            # Induction: Attention from 'is' to 'X' and 'Y'
            i_mass = attn[:, last_tok_idx, x_idx] + attn[:, last_tok_idx, y_idx]
            induction_masses[l] += i_mass
            
    retrieval_masses /= valid_count
    induction_masses /= valid_count
    
    flat_r = retrieval_masses.flatten().cpu().numpy()
    flat_i = induction_masses.flatten().cpu().numpy()
    
    # Top 5 Retrieval Heads
    top_r_idx = np.argsort(flat_r)[::-1][:5]
    retrieval_heads = [(int(idx // num_heads), int(idx % num_heads)) for idx in top_r_idx]
    
    # Top 5 Induction/Counting Heads
    top_i_idx = np.argsort(flat_i)[::-1][:5]
    induction_heads = [(int(idx // num_heads), int(idx % num_heads)) for idx in top_i_idx]
    
    print(f"\nIdentified Retrieval Heads: {retrieval_heads}")
    print(f"Identified Induction Heads: {induction_heads}")
    
    # Step 2: Measure Phase Activations
    tracker = PhaseActivationHook(num_heads, head_dim)
    tracker.register(model)
    
    for item in tqdm(dataset, desc="Measuring Phase Activations"):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            # max_new_tokens=2 ensures we get at least one forward pass where seq_len==1 (the decode phase)
            model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
            
    tracker.remove()
    
    # Step 3: Analyze Dominance
    def get_avg_norm(heads, phase):
        total = 0
        count = 0
        for h in heads:
            norms = tracker.norms[h][phase]
            total += sum(norms)
            count += len(norms)
        return total / count if count > 0 else 0
        
    retrieval_prefill = get_avg_norm(retrieval_heads, "prefill")
    retrieval_decode = get_avg_norm(retrieval_heads, "decode")
    
    induction_prefill = get_avg_norm(induction_heads, "prefill")
    induction_decode = get_avg_norm(induction_heads, "decode")
    
    print("\n=== EXPERIMENT 0 RESULTS ===")
    print(f"Retrieval Heads    | Prefill Norm: {retrieval_prefill:.4f} | Decode Norm: {retrieval_decode:.4f}")
    print(f"Induction Heads    | Prefill Norm: {induction_prefill:.4f} | Decode Norm: {induction_decode:.4f}")
    
    r_ratio = retrieval_prefill / (retrieval_decode + 1e-9)
    i_ratio = induction_decode / (induction_prefill + 1e-9)
    
    print(f"\nRetrieval Prefill/Decode Ratio: {r_ratio:.2f}x")
    print(f"Induction Decode/Prefill Ratio: {i_ratio:.2f}x")
    
    passed = (r_ratio > 1.0) and (i_ratio > 1.0)
    print(f"\nHARD GATE PASSED: {passed}")
    
    results = {
        "retrieval_heads": retrieval_heads,
        "induction_heads": induction_heads,
        "metrics": {
            "retrieval_prefill_norm": retrieval_prefill,
            "retrieval_decode_norm": retrieval_decode,
            "induction_prefill_norm": induction_prefill,
            "induction_decode_norm": induction_decode,
        },
        "gate_passed": passed
    }
    
    with open(os.path.join(OUTPUT_DIR, f"exp0_temporal_handoff_{model_key}.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment_0("qwen-0.5b")
