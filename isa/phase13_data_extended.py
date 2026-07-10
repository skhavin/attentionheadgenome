import json
import random

def generate_comparison():
    prompts = []
    # Generate 45 prompts comparing two numbers
    for i in range(45):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        while a == b:
            b = random.randint(10, 99)
        larger = max(a, b)
        prompt = f"Which is larger, {a} or {b}? "
        prompts.append({
            "id": f"comp_{i}",
            "task_type": "comparison",
            "prompt": prompt,
            "target": str(larger)
        })
    return prompts

def generate_sorting():
    prompts = []
    words = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon",
             "Mango", "Nectarine", "Orange", "Papaya", "Quince", "Raspberry", "Strawberry", "Tangerine", "Ugli", "Vanilla",
             "Watermelon", "Xigua", "Yam", "Zucchini", "Carrot", "Broccoli", "Spinach", "Potato", "Tomato", "Onion"]
    for i in range(45):
        sample = random.sample(words, 3)
        sorted_sample = sorted(sample)
        prompt = f"Sort alphabetically: {sample[0]}, {sample[1]}, {sample[2]} ->"
        # We target the first word of the sorted list to make it a single token prediction for DLA/residual extraction stability
        prompts.append({
            "id": f"sort_{i}",
            "task_type": "sorting",
            "prompt": prompt,
            "target": " " + sorted_sample[0]
        })
    return prompts

def generate_arithmetic():
    prompts = []
    for i in range(45):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        target = a + b
        prompt = f"The sum of {a} and {b} is"
        prompts.append({
            "id": f"arith_{i}",
            "task_type": "arithmetic",
            "prompt": prompt,
            "target": " " + str(target)
        })
    return prompts

def main():
    with open("dataset_discovery_140.json", "r", encoding="utf-8") as f:
        discovery = json.load(f)
        
    with open("dataset_confirmation_70.json", "r", encoding="utf-8") as f:
        confirmation = json.load(f)
        
    all_data = []
    all_data.extend(generate_comparison()[:42])
    all_data.extend(generate_sorting()[:42])
    all_data.extend(generate_arithmetic()[:42])
    
    by_task = {}
    for item in all_data:
        by_task.setdefault(item["task_type"], []).append(item)
        
    for task, items in by_task.items():
        discovery.extend(items[:28])
        confirmation.extend(items[28:42])
        
    random.shuffle(discovery)
    random.shuffle(confirmation)
    
    with open("dataset_discovery_224.json", "w", encoding="utf-8") as f:
        json.dump(discovery, f, indent=2)
        
    with open("dataset_confirmation_112.json", "w", encoding="utf-8") as f:
        json.dump(confirmation, f, indent=2)
        
    print(f"Extended Discovery Set: {len(discovery)} prompts (8 categories)")
    print(f"Extended Confirmation Set: {len(confirmation)} prompts (8 categories)")

if __name__ == "__main__":
    main()
