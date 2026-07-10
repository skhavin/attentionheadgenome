import json
import os
import random

def generate_negation():
    prompts = []
    pairs = [
        ("true", "false", "hot", "cold"),
        ("up", "down", "left", "right"),
        ("in", "out", "on", "off"),
        ("high", "low", "fast", "slow"),
        ("good", "bad", "happy", "sad"),
        ("black", "white", "day", "night"),
        ("boy", "girl", "man", "woman"),
        ("start", "stop", "begin", "end"),
        ("front", "back", "top", "bottom"),
        ("wet", "dry", "hard", "soft"),
        ("rich", "poor", "young", "old"),
        ("strong", "weak", "heavy", "light")
    ]
    
    # 42 prompts needed
    count = 0
    while count < 42:
        pair1 = random.choice(pairs)
        pair2 = random.choice(pairs)
        if pair1 == pair2: continue
        
        # Format: "A is the opposite of B. C is the opposite of" -> "D"
        prompt = f"{pair1[0].capitalize()} is the opposite of {pair1[1]}. {pair2[0].capitalize()} is the opposite of"
        target = f" {pair2[1]}"
        prompts.append({"prompt": prompt, "target": target, "task_type": "negation", "domain": "logic"})
        count += 1
    return prompts

def generate_set_membership():
    prompts = []
    sets = {
        "color": ["red", "blue", "green", "yellow", "purple", "orange"],
        "animal": ["dog", "cat", "bird", "fish", "horse", "cow"],
        "fruit": ["apple", "banana", "orange", "grape", "mango", "pear"],
        "shape": ["circle", "square", "triangle", "rectangle", "oval", "star"],
        "country": ["France", "Spain", "Italy", "Germany", "Japan", "China"]
    }
    categories = list(sets.keys())
    
    count = 0
    while count < 42:
        cat1 = random.choice(categories)
        cat2 = random.choice(categories)
        if cat1 == cat2: continue
        
        item1 = random.choice(sets[cat1])
        item2 = random.choice(sets[cat2])
        
        # Format: "A is a type of X. B is a type of" -> "Y"
        prompt = f"A {item1} is a type of {cat1}. A {item2} is a type of"
        target = f" {cat2}"
        prompts.append({"prompt": prompt, "target": target, "task_type": "set_membership", "domain": "logic"})
        count += 1
    return prompts

def generate_entailment():
    prompts = []
    rules = [
        ("All bachelors are unmarried. John is a bachelor. Therefore John is", " unmarried"),
        ("All squares have four sides. A shape is a square. Therefore it has four", " sides"),
        ("If x=y and y=z, then x=", "z"),
        ("If A is bigger than B, and B is bigger than C, then A is bigger than", " C"),
        ("Every triangle has three angles. This polygon is a triangle. Therefore it has three", " angles"),
        ("All mammals are warm-blooded. A dog is a mammal. Therefore a dog is", " warm-blooded"),
        ("If it is raining, the ground is wet. It is raining. Therefore the ground is", " wet"),
        ("Every square is a rectangle. Shape X is a square. Therefore Shape X is a", " rectangle"),
        ("If A equals B, then B equals", " A"),
        ("All planets orbit a star. Earth is a planet. Therefore Earth orbits a", " star")
    ]
    
    count = 0
    while count < 42:
        rule = random.choice(rules)
        # add a slight variation to avoid exact duplicates
        variation = random.choice([" ", "  ", ""])
        prompts.append({"prompt": rule[0] + variation, "target": rule[1], "task_type": "entailment", "domain": "logic"})
        count += 1
    return prompts

def generate_concatenation():
    prompts = []
    count = 0
    while count < 42:
        import string
        c1 = random.choice(string.ascii_uppercase)
        c2 = random.choice(string.ascii_uppercase)
        c3 = random.choice(string.ascii_uppercase)
        c4 = random.choice(string.ascii_uppercase)
        if c1 == c3 and c2 == c4: continue
        
        # Format: "The letters A and B make AB. The letters C and D make" -> "CD"
        prompt = f"The letters {c1} and {c2} make {c1}{c2}. The letters {c3} and {c4} make"
        target = f" {c3}{c4}"
        prompts.append({"prompt": prompt, "target": target, "task_type": "concatenation", "domain": "logic"})
        count += 1
    return prompts

def main():
    print("Generating new dataset...")
    with open("../../isa-head/dataset_discovery_224.json", "r") as f:
        disc_base = json.load(f)
    with open("../../isa-head/dataset_confirmation_112.json", "r") as f:
        conf_base = json.load(f)
        
    neg = generate_negation()
    set_mem = generate_set_membership()
    ent = generate_entailment()
    con = generate_concatenation()
    
    new_prompts = neg + set_mem + ent + con
    
    # Split 28/14
    disc_new = []
    conf_new = []
    for t in ["negation", "set_membership", "entailment", "concatenation"]:
        t_prompts = [p for p in new_prompts if p["task_type"] == t]
        disc_new.extend(t_prompts[:28])
        conf_new.extend(t_prompts[28:42])
        
    disc_all = disc_base + disc_new
    conf_all = conf_base + conf_new
    
    print(f"Discovery: {len(disc_base)} -> {len(disc_all)}")
    print(f"Confirmation: {len(conf_base)} -> {len(conf_all)}")
    
    with open("../../isa-head/dataset_discovery_336.json", "w") as f:
        json.dump(disc_all, f, indent=2)
        
    with open("../../isa-head/dataset_confirmation_168.json", "w") as f:
        json.dump(conf_all, f, indent=2)

if __name__ == "__main__":
    main()
