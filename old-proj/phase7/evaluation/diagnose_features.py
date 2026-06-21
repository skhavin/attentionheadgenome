import os
import sys
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase7.moe.moe_patcher import MoEPatcher
from phase7.evaluation.eval_ruler_advanced import make_vt_sample

def print_stats(name, stats):
    if stats is None:
        print(f"{name}: No stats collected.")
        return
    features = ["LocalEntropy", "SinkMass", "RecencyMass", "MaxSim"]
    print(f"\n--- {name} Feature Distribution ---")
    print(f"{'Feature':<15} | {'Min':>8} | {'Max':>8} | {'Mean':>8}")
    print("-" * 45)
    for i, f in enumerate(features):
        print(f"{f:<15} | {stats['min'][i]:8.3f} | {stats['max'][i]:8.3f} | {stats['mean'][i]:8.3f}")

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    checkpoint_path = "checkpoints/latest-qwen-fixed/stage2_routers.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    patcher = MoEPatcher(model)
    if os.path.exists(checkpoint_path):
        print(f"Loading routers from {checkpoint_path}")
        routers_dict = torch.load(checkpoint_path, map_location=device)
        for name, state_dict in routers_dict.items():
            patcher.routers[name].load_state_dict(state_dict)
    else:
        print("Warning: router checkpoint not found. Testing with random initialization.")

    # 1. Wikitext Distribution (Stage 1/2 training proxy)
    print("\nRunning Wikitext inference...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(ds["text"][:100]), return_tensors="pt")
    input_ids = encodings.input_ids[:, :2048].to(device)

    patcher.reset_activation_stats()
    patcher.hard_routing = True # So it uses the router forward
    with torch.no_grad():
        model(input_ids)

    wiki_stats = patcher.get_feature_stats()
    wiki_activations = patcher.get_activation_stats()
    print_stats("Wikitext", wiki_stats)
    print(f"Wikitext Path Activations: Sink={wiki_activations[0]:.1f}%, Local={wiki_activations[1]:.1f}%, Rec={wiki_activations[2]:.1f}%, Full={wiki_activations[3]:.1f}%")

    # 2. RULER (Variable Tracking) Distribution
    print("\nRunning RULER (Variable Tracking) inference...")
    rng = random.Random(42)
    sample = make_vt_sample(tokenizer, 2048, rng)
    ruler_inputs = tokenizer(sample["text"], return_tensors="pt").to(device)

    patcher.reset_activation_stats()
    with torch.no_grad():
        model(ruler_inputs.input_ids)

    ruler_stats = patcher.get_feature_stats()
    ruler_activations = patcher.get_activation_stats()
    print_stats("RULER (Variable Tracking)", ruler_stats)
    print(f"RULER Path Activations: Sink={ruler_activations[0]:.1f}%, Local={ruler_activations[1]:.1f}%, Rec={ruler_activations[2]:.1f}%, Full={ruler_activations[3]:.1f}%")

if __name__ == "__main__":
    main()
