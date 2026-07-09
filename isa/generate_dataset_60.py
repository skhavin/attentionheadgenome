import json

def generate():
    data = []
    
    # 20 Fact Recall Prompts (Country -> Capital)
    countries = ["France", "Germany", "Japan", "Italy", "Spain", "Canada", "China", "India", "Brazil", "Mexico",
                 "Russia", "Australia", "Egypt", "Argentina", "Sweden", "Norway", "Finland", "Peru", "Chile", "Greece"]
    capitals = [" Paris", " Berlin", " Tokyo", " Rome", " Madrid", " Ottawa", " Beijing", " New Delhi", " Brasília", " Mexico City",
                " Moscow", " Canberra", " Cairo", " Buenos Aires", " Stockholm", " Oslo", " Helsinki", " Lima", " Santiago", " Athens"]
    
    # We use a 1-shot prompt to set the task
    for i in range(20):
        country = countries[i]
        target = capitals[i]
        # Base country is UK (London)
        prompt = f"The capital of the United Kingdom is London. The capital of {country} is"
        data.append({"id": f"fact_{i:02d}", "task_type": "fact_recall", "prompt": prompt, "target": target})

    # 20 Pattern Induction Prompts (A -> B translation)
    # Mapping numbers to letters or simple repeated pairs
    # E.g. 1: A, 2: B. 3:
    for i in range(20):
        # We will use simple arithmetic mapping strings
        prompt = f"1: A, 2: B, 3: C. {i+4}:"
        target = f" {chr(65 + (i+3) % 26)}" # simple loop
        data.append({"id": f"induction_{i:02d}", "task_type": "pattern_induction", "prompt": prompt, "target": target})
        
    # 20 Needle In A Haystack (NIAH)
    # We will use varying passwords and varying junk text lengths
    passwords = [f"PW_{i*73}X" for i in range(20)]
    junk_base = "The study of artificial intelligence has progressed rapidly over the past decade, with significant advancements in natural language processing and computer vision. Machine learning models are now capable of performing complex tasks that were once thought impossible for computers. "
    
    for i in range(20):
        pw = passwords[i]
        junk = junk_base * (1 + (i % 3)) # varying length junk
        prompt = f"The secret password to unlock the matrix is '{pw}'. {junk}To proceed, please provide the secret password: The secret password to unlock the matrix is"
        # We expect the model to predict the first token of the password, which is usually " '" or " 'P"
        target = f" '{pw[:2]}" # Just an approximation for target verification, we will check if generated token matches start of pw
        data.append({
            "id": f"niah_{i:02d}", 
            "task_type": "niah", 
            "prompt": prompt, 
            "target_full": f" '{pw}'", 
            "password": pw
        })

    with open("dataset_60.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(data)} prompts.")

if __name__ == "__main__":
    generate()
