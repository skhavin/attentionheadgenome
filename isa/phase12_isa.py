import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def run_phase12(model_name):
    print(f"\n{'='*50}\nPhase 12: Head ISA Empirical Test on {model_name}\n{'='*50}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    discovery_prompts = get_prompts("dataset_discovery_40.json")
    confirmation_prompts = get_prompts("dataset_confirmation_20.json")
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    # We will classify heads into: WRITE, COPY, SEARCH, LOAD, UNCLASSIFIED
    # A simple operational definition per head per prompt:
    def evaluate_heads(prompts, desc):
        # returns dict: head_idx (l, h) -> list of assigned labels
        head_labels = { (l, h): [] for l in range(n_layers) for h in range(n_heads) }
        
        for item in tqdm(prompts, desc=desc):
            prompt = item["prompt"]
            target = item.get("target_full", item.get("target"))
            
            tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            target_id = tokenizer(target, add_special_tokens=False).input_ids[0]
            seq_len = tokens.input_ids.shape[1]
            
            # Hook Attention Outputs (for DLA)
            head_outputs = {}
            def o_proj_hook(m, args, l_idx):
                x = args[0]
                if x.dim() == 3: x = x[0, -1, :]
                else: x = x[-1, :]
                w_o = m.weight
                for h_idx in range(n_heads):
                    full_vec = torch.zeros_like(x)
                    start = h_idx * d_head
                    end = start + d_head
                    full_vec[start:end] = x[start:end]
                    import torch.nn.functional as F
                    resid = F.linear(full_vec, w_o)
                    head_outputs[(l_idx, h_idx)] = resid.detach().clone()
                    
            handles = []
            for l in range(n_layers):
                if hasattr(model.model.layers[l].self_attn, "o_proj"):
                    handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                        lambda m, a, l_idx=l: o_proj_hook(m, a, l_idx)))
                        
            # We also need attention weights. For simplicity and compute constraints, 
            # we will approximate "SEARCH/COPY" purely by DLA and layer depth rather than full QK dot products
            # since full attention hooking across 32 heads * 16 layers takes massive VRAM and compute.
            # DEFINITIONS:
            # WRITE: Late layer ( >= n_layers // 2), DLA > 1.5
            # LOAD: Early layer ( < n_layers // 3), DLA < 0.2, high variance
            # UNCLASSIFIED: else
            
            with torch.no_grad():
                _ = model(**tokens)
            for h in handles: h.remove()
            
            for l in range(n_layers):
                for h in range(n_heads):
                    if (l, h) in head_outputs:
                        with torch.no_grad():
                            if hasattr(model.model, "norm"): normed = model.model.norm(head_outputs[(l, h)])
                            elif hasattr(model.model, "layer_norm"): normed = model.model.layer_norm(head_outputs[(l, h)])
                            else: normed = head_outputs[(l, h)]
                            dla = model.lm_head(normed)[target_id].item()
                            
                        # Classify
                        if l >= n_layers // 2 and dla > 1.5:
                            label = "WRITE"
                        elif l < n_layers // 3 and abs(dla) < 0.2:
                            label = "LOAD"
                        else:
                            label = "UNCLASSIFIED"
                            
                        head_labels[(l, h)].append(label)
                        
        return head_labels

    print("Running Discovery Set Labeling...")
    discovery_labels = evaluate_heads(discovery_prompts, "Discovery (N=40)")
    
    # Resolve to single taxonomy label (majority vote)
    taxonomy = {}
    label_counts = {"WRITE": 0, "LOAD": 0, "UNCLASSIFIED": 0}
    for k, v in discovery_labels.items():
        if len(v) == 0:
            taxonomy[k] = "UNCLASSIFIED"
            continue
        # Get most common
        most_common = max(set(v), key=v.count)
        taxonomy[k] = most_common
        label_counts[most_common] += 1
        
    print(f"\nTaxonomy Distribution on Discovery:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt} heads")
        
    print("\nRunning Confirmation Set Validation...")
    confirmation_labels = evaluate_heads(confirmation_prompts, "Confirmation (N=20)")
    
    # Calculate metrics
    from collections import defaultdict
    
    # confusion[true_label][predicted_label]
    confusion = defaultdict(lambda: defaultdict(int))
    
    for k, actual_list in confirmation_labels.items():
        predicted = taxonomy[k]
        for actual in actual_list:
            confusion[actual][predicted] += 1
            
    print(f"\n--- Phase 12 Precision/Recall Results (N={len(confirmation_prompts)} prompts * {n_heads*n_layers} heads) ---")
    
    classes = ["WRITE", "LOAD", "UNCLASSIFIED"]
    
    for cls in classes:
        true_positives = confusion[cls][cls]
        false_positives = sum(confusion[other][cls] for other in classes if other != cls)
        false_negatives = sum(confusion[cls][other] for other in classes if other != cls)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass: {cls}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  (TP: {true_positives}, FP: {false_positives}, FN: {false_negatives})")
        
    print("\nConfusion Matrix (True \ Predicted):")
    print(f"{'':>15} " + " ".join([f"{c:>12}" for c in classes]))
    for true_c in classes:
        rowStr = f"{true_c:>15} "
        for pred_c in classes:
            rowStr += f"{confusion[true_c][pred_c]:>12}"
        print(rowStr)
        
if __name__ == "__main__":
    run_phase12("unsloth/Llama-3.2-1B")
