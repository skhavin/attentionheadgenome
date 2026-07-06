import json
import os
import random
import uuid
from datasets import load_dataset

DATA_DIR = "headgenome2_circuits/datasets"
os.makedirs(DATA_DIR, exist_ok=True)
N = 50  # 50 prompts per dataset as requested

def generate_uuid_dataset():
    """Generates synthetic UUID contexts for the Universal Copy Circuit test."""
    data = []
    for _ in range(N):
        target_uuid = str(uuid.uuid4())
        context_uuid = str(uuid.uuid4()) # Distractor
        # We place the target UUID early, then some text, then ask to repeat it.
        prompt = f"User Request ID: {target_uuid}\nSession ID: {context_uuid}\nStatus: Active\nSystem: Please confirm the User Request ID to proceed.\nConfirmation ID: "
        data.append({
            "prompt": prompt,
            "target": target_uuid
        })
    with open(os.path.join(DATA_DIR, "copy_uuids.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {N} UUID copying prompts.")

def build_counting():
    data = []
    fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon", "Mango", "Nectarine", "Orange", "Papaya", "Quince", "Raspberry", "Strawberry", "Tangerine", "Ugli", "Watermelon"]
    
    # Generate 500 prompts
    for _ in range(500):
        length = random.randint(3, 15)
        selected = random.sample(fruits, length)
        
        prompt = "Inventory List:\n"
        for i, f in enumerate(selected):
            prompt += f"{i+1}. {f}\n"
            
        prompt += "\nThe total number of items in the list above is "
        data.append({
            "prompt": prompt,
            "target": str(length),
            "count": length
        })
        
    with open("headgenome2_circuits/datasets/counting.json", "w") as f:
        json.dump(data, f, indent=2)

def build_json_brackets():
    data = []
    # Make JSON nesting much deeper (5-7 levels) and add distractors to avoid ceiling effect
    for i in range(100):
        depth = random.randint(5, 7)
        prompt = f'{{"payload_{i}": '
        closures = "}"
        for d in range(depth):
            key_name = f"level_{d}"
            prompt += f'{{"{key_name}": '
            closures += "}"
            
        # Add some arrays and distractors inside the deepest level
        prompt += f'{{"id": {random.randint(1000,9999)}, "tags": ["tag1", "tag2"], "nested_array": [1, 2, 3], "data": "value"'
        closures += "}"
        
        data.append({
            "prompt": prompt,
            "target_closures": closures
        })
        
    with open("headgenome2_circuits/datasets/json_brackets.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {N} JSON structural prompts.")

def generate_wikitext_dataset():
    """Extracts exactly N random snippets from Wikitext-2 for general baseline testing."""
    print("Loading Wikitext-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter out empty lines or very short headings
    valid_texts = [t for t in ds['text'] if len(t.strip()) > 100]
    
    selected_texts = random.sample(valid_texts, N)
    data = [{"text": t.strip()} for t in selected_texts]
    
    with open(os.path.join(DATA_DIR, "wikitext_random.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated {N} Wikitext snippets.")

if __name__ == "__main__":
    random.seed(42) # For reproducibility
    generate_uuid_dataset()
    build_counting()
    build_json_brackets()
    generate_wikitext_dataset()
    print(f"All datasets generated in {DATA_DIR}")
