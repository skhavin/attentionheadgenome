import json
import random

def generate_fact_recall():
    facts = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"), ("Italy", "Rome"), 
        ("Spain", "Madrid"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("China", "Beijing"),
        ("India", "New Delhi"), ("Brazil", "Brasilia"), ("Russia", "Moscow"), ("UK", "London"),
        ("Egypt", "Cairo"), ("Mexico", "Mexico City"), ("South Korea", "Seoul"), ("Argentina", "Buenos Aires"),
        ("Turkey", "Ankara"), ("Greece", "Athens"), ("Thailand", "Bangkok"), ("Sweden", "Stockholm"),
        ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Switzerland", "Bern"),
        ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Peru", "Lima"), ("Chile", "Santiago"),
        ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Cuba", "Havana"), ("Vietnam", "Hanoi"),
        ("Malaysia", "Kuala Lumpur"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("New Zealand", "Wellington"),
        ("Kenya", "Nairobi"), ("South Africa", "Pretoria"), ("Nigeria", "Abuja"), ("Morocco", "Rabat")
    ]
    prompts = []
    for i, (country, capital) in enumerate(facts):
        # Sample two random contextual facts to build context
        context_facts = random.sample([f for f in facts if f[0] != country], 2)
        prompt = f"The capital of {context_facts[0][0]} is {context_facts[0][1]}. The capital of {context_facts[1][0]} is {context_facts[1][1]}. The capital of {country} is"
        prompts.append({
            "id": f"fact_{i}",
            "task_type": "fact_recall",
            "prompt": prompt,
            "target": " " + capital
        })
    return prompts

def generate_pattern_induction():
    patterns = []
    # Generate A B C A B C ... patterns
    for i in range(45):
        # Pick 3 random uppercase letters
        letters = random.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 3)
        prompt = f"{letters[0]} {letters[1]} {letters[2]} {letters[0]} {letters[1]} {letters[2]} {letters[0]} {letters[1]}"
        patterns.append({
            "id": f"pattern_{i}",
            "task_type": "pattern_induction",
            "prompt": prompt,
            "target": " " + letters[2]
        })
    return patterns

def generate_niah():
    prompts = []
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]
    animals = ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "chicken"]
    for i in range(45):
        code = str(random.randint(10, 99))
        filler1 = f"The {random.choice(animals)} is {random.choice(colors)}."
        filler2 = f"The {random.choice(animals)} is {random.choice(colors)}."
        prompt = f"The secret code is {code}. {filler1} {filler2} The secret code is"
        prompts.append({
            "id": f"niah_{i}",
            "task_type": "niah",
            "prompt": prompt,
            "target": " " + code
        })
    return prompts

def generate_copy():
    prompts = []
    words = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon",
             "Mango", "Nectarine", "Orange", "Papaya", "Quince", "Raspberry", "Strawberry", "Tangerine", "Ugli", "Vanilla",
             "Watermelon", "Xigua", "Yam", "Zucchini", "Carrot", "Broccoli", "Spinach", "Potato", "Tomato", "Onion",
             "Garlic", "Ginger", "Pepper", "Salt", "Sugar", "Flour", "Butter", "Cheese", "Milk", "Egg",
             "Bread", "Rice", "Pasta", "Bean", "Lentil", "Pea", "Corn", "Wheat", "Oat", "Barley"]
    for i in range(45):
        word = words[i]
        filler_word = random.choice([w for w in words if w != word])
        prompt = f"Repeat exactly: '{filler_word}'. Repeat exactly: '{word}'"
        # We target the exact string it should output (without the quote since the model might generate just the word)
        # Actually it might generate the word and then a quote. Let's make the prompt simpler.
        prompt = f"Input: {filler_word} -> Output: {filler_word}\nInput: {word} -> Output:"
        prompts.append({
            "id": f"copy_{i}",
            "task_type": "copy",
            "prompt": prompt,
            "target": " " + word
        })
    return prompts

def generate_counting():
    prompts = []
    numbers = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
               "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
               "Twenty-one", "Twenty-two", "Twenty-three", "Twenty-four", "Twenty-five", "Twenty-six", "Twenty-seven", "Twenty-eight", "Twenty-nine", "Thirty",
               "Thirty-one", "Thirty-two", "Thirty-three", "Thirty-four", "Thirty-five", "Thirty-six", "Thirty-seven", "Thirty-eight", "Thirty-nine", "Forty",
               "Forty-one", "Forty-two", "Forty-three", "Forty-four", "Forty-five"]
               
    for i in range(42):
        start_idx = random.randint(0, len(numbers) - 5)
        seq = numbers[start_idx:start_idx+4]
        target = numbers[start_idx+4]
        prompt = f"{seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},"
        prompts.append({
            "id": f"count_{i}",
            "task_type": "counting",
            "prompt": prompt,
            "target": " " + target
        })
    return prompts

def main():
    # Generate 42 prompts per category (28 for Discovery, 14 for Confirmation)
    all_data = []
    all_data.extend(generate_fact_recall()[:42])
    all_data.extend(generate_pattern_induction()[:42])
    all_data.extend(generate_niah()[:42])
    all_data.extend(generate_copy()[:42])
    all_data.extend(generate_counting()[:42])
    
    # Split into Discovery (28 per category) and Confirmation (14 per category)
    discovery = []
    confirmation = []
    
    # Group by task_type
    by_task = {}
    for item in all_data:
        by_task.setdefault(item["task_type"], []).append(item)
        
    for task, items in by_task.items():
        discovery.extend(items[:28])
        confirmation.extend(items[28:42])
        
    random.shuffle(discovery)
    random.shuffle(confirmation)
    
    with open("dataset_discovery_140.json", "w", encoding="utf-8") as f:
        json.dump(discovery, f, indent=2)
        
    with open("dataset_confirmation_70.json", "w", encoding="utf-8") as f:
        json.dump(confirmation, f, indent=2)
        
    print(f"Generated Discovery Set: {len(discovery)} prompts")
    print(f"Generated Confirmation Set: {len(confirmation)} prompts")

if __name__ == "__main__":
    main()
