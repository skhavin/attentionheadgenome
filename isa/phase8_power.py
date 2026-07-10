import json
import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_power(mean_bc, std_bc, mean_n, std_n, n_samples=20, n_simulations=5000, alpha=0.05):
    sig_count = 0
    # Handle zero variance edge cases
    if std_bc == 0: std_bc = 1e-6
    if std_n == 0: std_n = 1e-6
        
    for _ in range(n_simulations):
        # Draw N samples for Boost-Correct group
        sample_bc = np.random.normal(loc=mean_bc, scale=std_bc, size=n_samples)
        # Draw N samples for Neutral group
        sample_n = np.random.normal(loc=mean_n, scale=std_n, size=n_samples)
        
        try:
            # Mann-Whitney U test (alternative='greater' means we test if BC is stochastically greater than Neutral)
            stat, p_val = stats.mannwhitneyu(sample_bc, sample_n, alternative='greater')
            if p_val < alpha:
                sig_count += 1
        except:
            pass
    return sig_count / n_simulations

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Loading model: {model_name} on {DEVICE} for Phase 8 Power Analysis")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    with open("dataset_discovery_40.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    n_layers = model.config.num_hidden_layers
    
    # Store ablation logit drops
    bc_drops = []
    neutral_drops = []
    
    DLA_THRESHOLD = 1.5 # threshold to be considered Boost-Correct

    for item in tqdm(prompts, desc="Evaluating Discovery Set MLPs"):
        tokens = tokenizer(item["prompt"], return_tensors="pt").to(DEVICE)
        target_token = item.get("target_full", item.get("target"))
        target_id = tokenizer(target_token, add_special_tokens=False).input_ids[0]
        
        # 1. Clean Run & Get MLP Outputs
        mlp_outputs = {}
        def mlp_save_hook(m, args, output, l_idx):
            mlp_outputs[l_idx] = output[0, -1, :].detach().clone()
            
        handles = []
        for l in range(n_layers):
            handles.append(model.model.layers[l].mlp.register_forward_hook(
                lambda m, a, o, l_idx=l: mlp_save_hook(m, a, o, l_idx)))
                
        with torch.no_grad():
            clean_logits = model(**tokens).logits
        for h in handles: h.remove()
        
        clean_target_logit = clean_logits[0, -1, target_id].item()
        
        # 2. DLA Categorization
        bc_layers = []
        neutral_layers = []
        
        for l in range(n_layers):
            # Calculate DLA
            with torch.no_grad():
                dla_logits = model.lm_head(model.model.norm(mlp_outputs[l]))
                target_dla = dla_logits[target_id].item()
                
            if target_dla > DLA_THRESHOLD:
                bc_layers.append(l)
            elif abs(target_dla) < 0.2: # strict neutral
                neutral_layers.append(l)
                
        # Sample max 2 from each category per prompt to keep compute bounded
        # We just need variance estimates, not an exhaustive ablation of all 28 layers per prompt
        test_bc = bc_layers[:2]
        test_n = neutral_layers[:2]
        
        # 3. Causal Ablation (Mean Ablation)
        def run_ablation(layer_to_ablate):
            def ablate_hook(m, args, output):
                # zero out the output at the last token
                output[0, -1, :] = 0.0
                return output
                
            h = model.model.layers[layer_to_ablate].mlp.register_forward_hook(ablate_hook)
            with torch.no_grad():
                abl_logits = model(**tokens).logits
            h.remove()
            return clean_target_logit - abl_logits[0, -1, target_id].item()
            
        for l in test_bc:
            bc_drops.append(run_ablation(l))
            
        for l in test_n:
            neutral_drops.append(run_ablation(l))
            
    print(f"\nCollected {len(bc_drops)} Boost-Correct ablations and {len(neutral_drops)} Neutral ablations.")
    
    if len(bc_drops) == 0:
        print("ERROR: No MLPs crossed the DLA threshold to be classified as Boost-Correct.")
        return
        
    mean_bc = np.mean(bc_drops)
    std_bc = np.std(bc_drops)
    
    mean_n = np.mean(neutral_drops)
    std_n = np.std(neutral_drops)
    
    print("\n=== Phase 8 Causal Ablation Effect Statistics ===")
    print(f"Boost-Correct MLPs -> Mean Logit Drop: {mean_bc:.3f} (Std: {std_bc:.3f})")
    print(f"Neutral MLPs       -> Mean Logit Drop: {mean_n:.3f} (Std: {std_n:.3f})")
    
    print("\nRunning Monte Carlo Power Analysis (N=20)...")
    power_20 = calculate_power(mean_bc, std_bc, mean_n, std_n, n_samples=20)
    print(f"Statistical Power at N=20: {power_20*100:.1f}%")
    
    if power_20 < 0.8:
        power_40 = calculate_power(mean_bc, std_bc, mean_n, std_n, n_samples=40)
        power_80 = calculate_power(mean_bc, std_bc, mean_n, std_n, n_samples=80)
        print(f"Statistical Power at N=40: {power_40*100:.1f}%")
        print(f"Statistical Power at N=80: {power_80*100:.1f}%")

if __name__ == "__main__":
    main()
