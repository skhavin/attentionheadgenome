import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import numpy as np
import scipy.stats

os.makedirs("outputs/phase8_paper_suite", exist_ok=True)

import sys

model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", output_attentions=True)
model.eval()

num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0))
num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 0))

# Prompt Groups
prompt_groups = {
    "PlainText": [
        "The history of the Roman Empire is long and complex. It began as a small city-state in Italy and grew to encompass the entire Mediterranean basin.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy to fuel their growth and activities.",
        "The Great Wall of China is one of the most recognizable structures in the world, stretching thousands of miles across the landscape."
    ],
    "Copy": [
        "The ID is 9f86d0. The system verified it. The ID is",
        "Her name is Sarah Connor. She was targeted by a cyborg. Her name is",
        "The pass code is 4829. Access granted. The pass code is"
    ],
    "Retrieval": [
        "The secret key is Alpha77. We walked down the street and bought some apples. We then went to the park. What is the secret key? It is",
        "The patient's blood type is O-negative. The doctor checked the charts and ordered some tests. The blood type is",
        "The artifact is hidden in the blue box. The room was dark and dusty. The artifact is hidden in the"
    ],
    "Code": [
        "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
        "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        pass",
        "import numpy as np\nx = np.array([1, 2, 3])\ny = np.sum(x)\nprint(y)"
    ],
    "JSON": [
        '{"user": {"id": 101, "name": "Alice", "roles": ["admin", "editor"]}, "status": "active"}',
        '{"config": {"host": "localhost", "port": 8080, "timeout": 30}, "retries": 5}',
        '{"menu": {"popup": {"menuitem": [{"value": "New", "onclick": "CreateNewDoc()"}]}}}'
    ],
    "Dialogue": [
        "User: Can you help me with Python?\nAssistant: Of course! What do you need help with?\nUser: How do I write a loop?\nAssistant:",
        "Alice: Hi Bob, how are you doing today?\nBob: I'm doing well, thanks. Did you finish the report?\nAlice: Yes, I sent it.",
        "Customer: I would like to order a pizza.\nAgent: Sure, what toppings would you like?\nCustomer: Pepperoni and mushrooms please."
    ],
    "Math": [
        "Let x = 5 and y = 10. Then x + y = 15. If we multiply by 2, we get 30.",
        "To solve 3x - 7 = 14, add 7 to both sides to get 3x = 21. Then divide by 3 to get x = 7.",
        "The area of a circle is pi * r^2. If r = 4, then the area is 16 * pi."
    ],
    "Repetition": [
        "A A A A A A A A A A A A A A A A A A A",
        "1010101010101010101010101010101010101",
        "test test test test test test test test test test test"
    ]
}

# Metric: Locality (attention mass in the last 5 tokens)
LOCAL_WINDOW = 5

head_behaviors = {l: {h: {group: [] for group in prompt_groups.keys()} for h in range(num_heads)} for l in range(num_layers)}

print("Evaluating prompts...")
with torch.no_grad():
    for group_name, prompts in prompt_groups.items():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            seq_len = inputs.input_ids.shape[1]
            
            # Skip if sequence is too short
            if seq_len <= LOCAL_WINDOW:
                continue
                
            outputs = model(**inputs)
            attentions = outputs.attentions # tuple of (batch, heads, seq, seq)
            
            # Look at attention from the very last token
            for l in range(num_layers):
                attn = attentions[l][0, :, -1, :] # (heads, seq)
                for h in range(num_heads):
                    # Locality: sum of attention mass in the last LOCAL_WINDOW tokens
                    locality = attn[h, -LOCAL_WINDOW:].sum().item()
                    head_behaviors[l][h][group_name].append(locality)

# Compute variance and stability
print("Computing stability...")
head_stats = []

for l in range(num_layers):
    for h in range(num_heads):
        group_means = {}
        for group_name in prompt_groups.keys():
            vals = head_behaviors[l][h][group_name]
            group_means[group_name] = np.mean(vals) if len(vals) > 0 else 0.0
            
        means = list(group_means.values())
        variance = np.var(means)
        
        head_stats.append({
            "layer": l,
            "head": h,
            "variance": variance,
            "group_means": group_means
        })

# Sort by variance
head_stats.sort(key=lambda x: x["variance"], reverse=True)

most_unstable = head_stats[:10]
most_stable = head_stats[-10:]

results = {
    "model": model_id,
    "metric": f"Locality (W={LOCAL_WINDOW})",
    "top_10_regime_switchers": most_unstable,
    "top_10_most_stable": most_stable
}

with open(f"outputs/phase8_paper_suite/regime_switching_{model_id.replace('/', '_')}.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Top 3 Regime-Switching Heads (Dynamic) ===")
for head in most_unstable[:3]:
    print(f"\nHead L{head['layer']}H{head['head']} (Variance: {head['variance']:.4f})")
    for group, val in head["group_means"].items():
        bar = "#" * int(val * 20)
        print(f"  {group[:10]:<10}: {val:.2f} | {bar}")
        
print("\n=== Top 3 Most Stable Heads (Static) ===")
for head in most_stable[:3]:
    print(f"\nHead L{head['layer']}H{head['head']} (Variance: {head['variance']:.4f})")
    for group, val in head["group_means"].items():
        bar = "#" * int(val * 20)
        print(f"  {group[:10]:<10}: {val:.2f} | {bar}")

print(f"\nDone. Saved to outputs/phase8_paper_suite/regime_switching_{model_id.replace('/', '_')}.json")
