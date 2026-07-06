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

def generate_counting_dataset():
    """Generates counting tasks for the Counting Circuit test."""
    data = []
    items = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon"]
    for _ in range(N):
        count = random.randint(3, 8)
        selected = random.sample(items, count)
        
        # e.g., "1. Apple\n2. Banana\n3. Cherry\nTotal items:"
        list_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(selected)])
        prompt = f"Inventory List:\n{list_str}\n\nThe total number of items in the list above is "
        data.append({
            "prompt": prompt,
            "target": str(count),
            "count": count
        })
    with open(os.path.join(DATA_DIR, "counting.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {N} counting prompts.")

def generate_json_dataset():
    """Generates deeply nested JSON structure prompts for Structured Output test."""
    data = []
    for i in range(N):
        prompt = f'{{"user_{i}": {{"metadata": {{"id": {random.randint(1000,9999)}, "active": true, "roles": ["admin", "user"]'
        # Expecting the model to close the structures correctly: }}}, etc.
        data.append({
            "prompt": prompt,
            "target_closures": "}}}" # Just tracking what closing brackets are expected
        })
    with open(os.path.join(DATA_DIR, "json_brackets.json"), "w") as f:
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
    generate_counting_dataset()
    generate_json_dataset()
    generate_wikitext_dataset()
    print(f"All datasets generated in {DATA_DIR}")
