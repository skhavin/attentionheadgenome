import json
import os
import random

DATASET_DIR = "headgenome2_circuits/datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def generate_addition_dataset(num_samples=100):
    dataset = []
    
    # We want single digit sum for clean single-token output space (0-9)
    # Pairs of (X, Y) such that X + Y <= 9
    valid_pairs = []
    for x in range(1, 9):
        for y in range(1, 10 - x):
            valid_pairs.append((x, y))
            
    for _ in range(num_samples):
        x, y = random.choice(valid_pairs)
        prompt = f"Question: What is {x} plus {y}? Answer: The sum is"
        dataset.append({
            "prompt": prompt,
            "x": x,
            "y": y,
            "sum": x + y
        })
        
    out_path = os.path.join(DATASET_DIR, "arithmetic.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated arithmetic dataset to {out_path} with {len(dataset)} items.")

if __name__ == "__main__":
    generate_addition_dataset(200)
