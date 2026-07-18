import json
import os
import random
import numpy as np
from transformers import AutoTokenizer

def get_train_test_split(items, train_ratio=0.7):
    # Deterministic split for disjoint pools based on seed
    split_idx = int(len(items) * train_ratio)
    items_copy = list(items)
    random.shuffle(items_copy)
    return items_copy[:split_idx], items_copy[split_idx:]

def generate_category(cat_name, train_templates, test_templates, train_slots, test_slots, generator_func):
    train_prompts = []
    test_prompts = []
    
    # Generate 70 Train
    count = 0
    while count < 70:
        template = random.choice(train_templates)
        prompt, target = generator_func(template, train_slots)
        if not any(p["prompt"] == prompt for p in train_prompts):
            train_prompts.append({"prompt": prompt, "target": target, "task_type": cat_name, "domain": "mixed"})
            count += 1
            
    # Generate 30 Test
    count = 0
    while count < 30:
        template = random.choice(test_templates)
        prompt, target = generator_func(template, test_slots)
        if not any(p["prompt"] == prompt for p in test_prompts):
            test_prompts.append({"prompt": prompt, "target": target, "task_type": cat_name, "domain": "mixed"})
            count += 1
            
    return train_prompts, test_prompts

def get_comparison():
    train_t = [
        "The {a1} is {n1} years old. The {a2} is {n2} years old. The older animal is the",
        "Compare the ages: {a1} is {n1} and {a2} is {n2}. The oldest is",
        "Between {a1} at age {n1} and {a2} at age {n2}, the older one is",
        "An {a1} lived for {n1} years. An {a2} lived for {n2} years. Which lived longer? The",
        "Age comparison: {a1}({n1}), {a2}({n2}). The senior is the",
        "Consider two animals. First, a {a1} aged {n1}. Second, a {a2} aged {n2}. The one with the greater age is the",
        "If a {a1} is {n1} and a {a2} is {n2}, the older of the two is definitely the",
        "The {a1} reached {n1} years of age, while the {a2} reached {n2}. The older creature is the",
        "Given a {n1}-year-old {a1} and a {n2}-year-old {a2}, the oldest is the",
        "Which is older between a {a1} ({n1} yrs) and a {a2} ({n2} yrs)? The",
        "Evaluating longevity: {a1} is {n1}, {a2} is {n2}. The oldest is the",
        "The {a1} has an age of {n1}. The {a2} has an age of {n2}. The elder is the"
    ]
    test_t = [
        "Which is older? {a1} (age {n1}) or {a2} (age {n2})? Answer:",
        "If you have a {a1} that is {n1} years old and a {a2} that is {n2} years old, the senior animal is",
        "Determine the older creature: {a1} is {n1}, {a2} is {n2}. The oldest is"
    ]
    
    animals = ["dog", "cat", "lion", "tiger", "bear", "wolf", "eagle", "hawk", "shark", "dolphin", "elephant", "rhino", "whale", "squid", "falcon", "owl", "horse", "cow", "snake", "lizard"]
    train_animals, test_animals = get_train_test_split(animals)
    
    def gen_func(template, slots):
        a1, a2 = random.sample(slots["animals"], 2)
        n1, n2 = random.randint(10, 50), random.randint(1, 9)
        if random.random() < 0.5:
            n1, n2 = n2, n1
        ans = a1 if n1 > n2 else a2
        return template.format(a1=a1, a2=a2, n1=n1, n2=n2), f" {ans}"
        
    return generate_category("comparison", train_t, test_t, {"animals": train_animals}, {"animals": test_animals}, gen_func)

def get_copy():
    train_t = [
        "Repeat the word {w} exactly {n} times: {w} {w}",
        "Say the word '{w}' {n} times in a row: {w} {w}",
        "Echo this text: {w}, repeat it {n} times total: {w} {w}",
        "Output the following string exactly {n} times - {w}: {w} {w}",
        "Write {w} repeatedly. Give me exactly {n} of them: {w} {w}",
        "Print '{w}' {n} times consecutively. Output: {w} {w}",
        "Copy task: word is {w}, count is {n}. Go: {w} {w}",
        "Duplicate '{w}' until there are {n}. Start: {w} {w}",
        "Replicate the string {w} exactly {n} times: {w} {w}",
        "Please type {w} a total of {n} times: {w} {w}",
        "Generation request: {n} copies of {w}. Result: {w} {w}",
        "Produce the token '{w}' {n} times. Sequence: {w} {w}"
    ]
    test_t = [
        "Task: copy. Item: {w}. Times: {n}. Output: {w} {w}",
        "I need {n} exact copies of the word {w}: {w} {w}",
        "Just say {w} over and over, {n} times: {w} {w}"
    ]
    words = ["apple", "banana", "cat", "dog", "elephant", "frog", "guitar", "house", "igloo", "jungle", "kite", "lemon", "monkey", "ninja", "ocean", "piano", "queen", "river", "sun", "tree", "umbrella", "violin", "water", "xylophone", "yacht", "zebra"]
    train_w, test_w = get_train_test_split(words)
    
    def gen_func(template, slots):
        w = random.choice(slots["w"])
        n = random.randint(3, 9)
        return template.format(w=w, n=n), f" {w}"
        
    return generate_category("copy", train_t, test_t, {"w": train_w}, {"w": test_w}, gen_func)

def get_counting():
    train_t = [
        "I see {seq}. In total, I see",
        "Count these items: {seq}. Total count:",
        "There is {seq}. The number of items is",
        "List: {seq}. How many? Answer:",
        "We have {seq}. The total amount is",
        "Observe the following: {seq}. The final tally is",
        "Item enumeration: {seq}. Total equals",
        "Add them up: {seq}. The sum is",
        "Count the objects present: {seq}. Result:",
        "Quantity check for {seq}. The quantity is",
        "If you count {seq}, you get exactly",
        "Tallying items: {seq}. Final count is"
    ]
    test_t = [
        "How many are here? {seq}. Total:",
        "Count up the following array of items: {seq}. There are",
        "Evaluating the list: {seq}. Total number:"
    ]
    items = ["apples", "books", "cats", "dogs", "cars", "trees", "pens", "cups", "birds", "chairs", "desks", "phones", "keys", "shoes", "coins", "hats", "rings", "bags", "boxes", "stars"]
    train_items, test_items = get_train_test_split(items)
    word_map = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    
    def gen_func(template, slots):
        item = random.choice(slots["items"])
        n = random.randint(2, 6)
        seq = ", ".join([f"one {item}"] * n)
        return template.format(seq=seq), f" {word_map[n]}"
        
    return generate_category("counting", train_t, test_t, {"items": train_items}, {"items": test_items}, gen_func)

def get_fact_recall():
    train_t = [
        "The capital of {c} is",
        "If you travel to the capital of {c}, you are in",
        "{c}'s capital city is named",
        "The government of {c} is located in the capital,",
        "What is the capital of {c}? It is",
        "The primary capital city of {c} is",
        "Geographically, the capital of {c} is known as",
        "When visiting {c}, you might fly into its capital,",
        "The capital of the nation of {c} is",
        "Name the capital of {c}. The answer is",
        "The political center and capital of {c} is",
        "For the country {c}, the capital city is"
    ]
    test_t = [
        "Which city is the capital of {c}? Answer:",
        "The national capital of {c} is widely known to be",
        "Identify the capital city for {c}: it is",
        "Name the capital city of {c}:",
        "What is the capital city of {c}? Answer:",
        "The seat of government in {c} is located at"
    ]
    capitals = [
        ("France", "Paris"), ("Spain", "Madrid"), ("Italy", "Rome"), ("Germany", "Berlin"), 
        ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("India", "New"), 
        ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Egypt", "Cairo"),
        ("Mexico", "Mexico"), ("UK", "London"), ("USA", "Washington"), ("Peru", "Lima"),
        ("Chile", "Santiago"), ("Kenya", "Nairobi"), ("Thailand", "Bangkok"), ("Sweden", "Stockholm"),
        ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"),
        ("Greece", "Athens"), ("Turkey", "Ankara"), ("South Korea", "Seoul"), ("Vietnam", "Hanoi"),
        ("Argentina", "Buenos"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Cuba", "Havana"),
        ("Jamaica", "Kingston"), ("Morocco", "Rabat"), ("Nigeria", "Abuja"), ("South Africa", "Pretoria"),
        ("New Zealand", "Wellington"), ("Fiji", "Suva"), ("Portugal", "Lisbon"), ("Ireland", "Dublin")
    ]
    train_caps, test_caps = get_train_test_split(capitals)
    
    def gen_func(template, slots):
        c, cap = random.choice(slots["caps"])
        return template.format(c=c), f" {cap}"
        
    return generate_category("fact_recall", train_t, test_t, {"caps": train_caps}, {"caps": test_caps}, gen_func)

def get_sorting():
    train_t = [
        "Sort these numbers from smallest to largest: {n1}, {n2}, {n3} ->",
        "Order these from minimum to maximum: {n1}, {n2}, {n3}. The smallest is",
        "Given {n1}, {n2}, and {n3}, the lowest value is",
        "Put in ascending order: {n1}, {n2}, {n3}. It starts with",
        "Identify the minimum of the set [{n1}, {n2}, {n3}]:",
        "Sort ascending: {n1}, {n2}, {n3}. The first number is",
        "Arrange {n1}, {n2}, {n3} from low to high. The lowest is",
        "The smallest number out of {n1}, {n2}, and {n3} is",
        "Finding the minimum: {n1}, {n2}, {n3}. The answer is",
        "Which is smallest: {n1}, {n2}, or {n3}? It is",
        "Sort sequence {n1}, {n2}, {n3}. Minimum value:",
        "Rank by size, smallest first: {n1}, {n2}, {n3}. Starts with"
    ]
    test_t = [
        "Of the numbers {n1}, {n2}, and {n3}, the least is",
        "What is the minimum value among {n1}, {n2}, {n3}? Answer:",
        "Order ascending: {n1}, {n2}, {n3}. The sequence begins with"
    ]
    train_nums = list(range(1, 50))
    test_nums = list(range(51, 100))
    
    def gen_func(template, slots):
        nums = random.sample(slots["nums"], 3)
        return template.format(n1=nums[0], n2=nums[1], n3=nums[2]), f" {min(nums)}"
        
    return generate_category("sorting", train_t, test_t, {"nums": train_nums}, {"nums": test_nums}, gen_func)

def get_arithmetic():
    train_t = [
        "What is {a} {op} {b}? The answer is",
        "Calculate the result of {a} {op} {b}:",
        "Math problem: {a} {op} {b} =",
        "Solve this equation: {a} {op} {b}. Result:",
        "The sum or difference of {a} {op} {b} is",
        "Compute {a} {op} {b}. Answer:",
        "If you evaluate {a} {op} {b}, you get",
        "Arithmetic: {a} {op} {b}. Equals",
        "Find the value of {a} {op} {b}. It is",
        "Evaluate the expression {a} {op} {b} ->",
        "Solving {a} {op} {b} yields",
        "The mathematical result of {a} {op} {b} is"
    ]
    test_t = [
        "Please solve {a} {op} {b}. The value is",
        "What does {a} {op} {b} equal? Answer:",
        "Calculate {a} {op} {b} and output the result:"
    ]
    train_nums = list(range(2, 20))
    test_nums = list(range(21, 40))
    
    def gen_func(template, slots):
        a, b = random.choice(slots["nums"]), random.choice(slots["nums"])
        op = random.choice(["+", "-"])
        if op == "-" and a <= b:
            a, b = b, a
            if a == b: a += 1
        ans = a + b if op == "+" else a - b
        return template.format(a=a, b=b, op=op), f" {ans}"
        
    return generate_category("arithmetic", train_t, test_t, {"nums": train_nums}, {"nums": test_nums}, gen_func)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for split isolation")
    args = parser.parse_args()
    
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    
    print(f"Generating Redesigned Trajectory Dataset (Seed {args.split_seed})...")
    
    generators = [
        get_comparison, get_copy, get_counting, 
        get_fact_recall, get_sorting, get_arithmetic
    ]
    
    mapping = []
    validation = []
    
    for gen in generators:
        tr, te = gen()
        mapping.extend(tr)
        validation.extend(te)
        
    os.makedirs("../outputs/dataset", exist_ok=True)
    with open("../outputs/dataset/trajectory_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
        
    with open("../outputs/dataset/trajectory_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
        
    print(f"Generated {len(mapping)} mapping prompts (70/cat).")
    print(f"Generated {len(validation)} validation prompts (30/cat).")
    
    # Calculate length distributions
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    
    all_prompts = mapping + validation
    lengths_by_cat = {c: [] for c in ["comparison", "copy", "counting", "fact_recall", "sorting", "arithmetic"]}
    for p in all_prompts:
        toks = tokenizer.encode(p["prompt"])
        lengths_by_cat[p["task_type"]].append(len(toks))
        
    print("\n--- Length Distributions (Tokens) ---")
    length_metadata = {}
    for c, lengths in lengths_by_cat.items():
        length_metadata[c] = {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths))
        }
        print(f"{c:15s} | Mean: {length_metadata[c]['mean']:.1f} | Std: {length_metadata[c]['std']:.1f} | Range: [{length_metadata[c]['min']}, {length_metadata[c]['max']}]")
        
    with open("../outputs/dataset/length_metadata.json", "w", encoding="utf-8") as f:
        json.dump(length_metadata, f, indent=2)

if __name__ == "__main__":
    main()
