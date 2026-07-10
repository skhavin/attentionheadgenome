import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
from tqdm import tqdm
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Punctuation and common generation prefixes
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])
for p in string.punctuation: STOP_WORDS.add(p)
STOP_WORDS.update(["", " ", " \n", "\n"])

def classify_token(token_str, target_str):
    t_clean = token_str.strip().lower()
    tgt_clean = target_str.strip().lower()
    if t_clean == tgt_clean or tgt_clean in t_clean or (t_clean in tgt_clean and len(t_clean) > 2):
        return "Answer"
    if t_clean in STOP_WORDS or not t_clean.isalpha():
        return "Grammar"
    return "Concept"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def run_model_analysis(model_name, prompts):
    print(f"\n{'='*60}\nEvaluating Model: {model_name}\n{'='*60}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    head_dlas = []
    mlp_dlas = []
    
    first_concept_layers = []
    first_answer_layers = []
    
    for item in tqdm(prompts, desc=f"Evaluating {model_name}"):
        prompt = item["prompt"]
        target = item.get("target_full", item.get("target"))
        
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        target_id = tokenizer(target, add_special_tokens=False).input_ids[0]
        
        # --- 1. Phase 9 Timeline (Logit Lens) ---
        resid_stream = {}
        def resid_hook(m, args, output, l_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3: resid_stream[l_idx] = hidden[0, -1, :].detach().clone()
            else: resid_stream[l_idx] = hidden[-1, :].detach().clone()
                
        handles = []
        for l in range(n_layers):
            handles.append(model.model.layers[l].register_forward_hook(
                lambda m, a, o, l_idx=l: resid_hook(m, a, o, l_idx)))
                
        with torch.no_grad():
            _ = model(**tokens)
        for h in handles: h.remove()
        
        trajectory = []
        for l in range(n_layers):
            with torch.no_grad():
                # Apply final layernorm equivalent
                if hasattr(model.model, "norm"):
                    normed = model.model.norm(resid_stream[l])
                elif hasattr(model.model, "layer_norm"):
                    normed = model.model.layer_norm(resid_stream[l])
                else:
                    normed = resid_stream[l]
                    
                logits = model.lm_head(normed)
                top_id = logits.argmax().item()
                top_str = tokenizer.decode(top_id)
                
            cat = classify_token(top_str, target)
            trajectory.append(cat)
            
        try: c_layer = trajectory.index("Concept")
        except ValueError: c_layer = n_layers
        try: a_layer = trajectory.index("Answer")
        except ValueError: a_layer = n_layers
        
        first_concept_layers.append(c_layer)
        first_answer_layers.append(a_layer)

        # --- 2. Phase 4 DLA Gap (Attention vs MLPs) ---
        # Hook MLPs
        mlp_outputs = {}
        def mlp_save_hook(m, args, output, l_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3: mlp_outputs[l_idx] = hidden[0, -1, :].detach().clone()
            else: mlp_outputs[l_idx] = hidden[-1, :].detach().clone()
            
        handles_mlp = []
        for l in range(n_layers):
            if hasattr(model.model.layers[l], "mlp"):
                handles_mlp.append(model.model.layers[l].mlp.register_forward_hook(
                    lambda m, a, o, l_idx=l: mlp_save_hook(m, a, o, l_idx)))
        
        with torch.no_grad():
            _ = model(**tokens)
        for h in handles_mlp: h.remove()
        
        late_mlp_dlas = []
        start_late = int(n_layers * 0.7)
        for l in range(start_late, n_layers):
            if l in mlp_outputs:
                with torch.no_grad():
                    if hasattr(model.model, "norm"): normed = model.model.norm(mlp_outputs[l])
                    elif hasattr(model.model, "layer_norm"): normed = model.model.layer_norm(mlp_outputs[l])
                    else: normed = mlp_outputs[l]
                    dla_logits = model.lm_head(normed)
                    late_mlp_dlas.append(dla_logits[target_id].item())
                    
        mlp_dlas.append(np.mean(late_mlp_dlas) if late_mlp_dlas else 0)

        # Hook Attention Heads
        head_outputs = {}
        def head_resid_save_hook(m, args, l_idx):
            x = args[0]
            if x.dim() == 3: x = x[0, -1, :]
            else: x = x[-1, :]
            
            w_o = m.weight
            for h_idx in range(n_heads):
                # Isolate the h_idx'th head vector
                full_vec = torch.zeros_like(x)
                start_dim = h_idx * d_head
                end_dim = start_dim + d_head
                full_vec[start_dim:end_dim] = x[start_dim:end_dim]
                
                import torch.nn.functional as F
                resid = F.linear(full_vec, w_o)
                head_outputs[(l_idx, h_idx)] = resid.detach().clone()
                
        handles_attn = []
        for l in range(n_layers):
            if hasattr(model.model.layers[l], "self_attn") and hasattr(model.model.layers[l].self_attn, "o_proj"):
                handles_attn.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, l_idx=l: head_resid_save_hook(m, a, l_idx)))
                    
        with torch.no_grad():
            _ = model(**tokens)
        for h in handles_attn: h.remove()
        
        all_head_dlas = []
        for l in range(n_layers):
            for h_idx in range(n_heads):
                if (l, h_idx) in head_outputs:
                    with torch.no_grad():
                        if hasattr(model.model, "norm"): normed = model.model.norm(head_outputs[(l, h_idx)])
                        elif hasattr(model.model, "layer_norm"): normed = model.model.layer_norm(head_outputs[(l, h_idx)])
                        else: normed = head_outputs[(l, h_idx)]
                        dla = model.lm_head(normed)[target_id].item()
                        all_head_dlas.append(dla)
                        
        all_head_dlas.sort(reverse=True)
        top_5_mean = np.mean(all_head_dlas[:5]) if all_head_dlas else 0
        head_dlas.append(top_5_mean)

    print(f"\n--- Phase 11 Scoped Results: {model_name} ---")
    print("1. Phase 9 Timeline:")
    print(f"Median First 'Concept' Layer: {np.median(first_concept_layers)}")
    print(f"Median First 'Answer' Layer:  {np.median(first_answer_layers)}")
    
    print("\n2. Phase 4 DLA Gap (Top 5 Heads vs Late MLPs):")
    mean_head_dla = np.mean(head_dlas)
    mean_mlp_dla = np.mean(mlp_dlas)
    print(f"Mean Top 5 Head DLA: {mean_head_dla:.3f}")
    print(f"Mean Late MLP DLA:   {mean_mlp_dla:.3f}")
    
    try:
        # We test if MLP DLA is significantly greater than Head DLA
        stat, p_val = stats.wilcoxon(mlp_dlas, head_dlas, alternative='greater')
        print(f"Wilcoxon p-value (MLP > Head): {p_val:.3e}")
        if p_val < 0.05 and mean_mlp_dla > mean_head_dla:
            print(">> FINDING HOLDS: MLPs dominate factual writing.")
        else:
            print(">> FINDING FAILS: MLPs do not dominate factual writing.")
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")

    # Free memory before next model
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    prompts = get_prompts("dataset_confirmation_20.json")
    print(f"Found {len(prompts)} prompts in Confirmation Set.")
    
    models_to_test = [
        "unsloth/Llama-3.2-1B"
    ]
    
    for m in models_to_test:
        run_model_analysis(m, prompts)

if __name__ == "__main__":
    main()
