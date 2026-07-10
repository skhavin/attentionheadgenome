import json
import random
import os

def main():
    with open("dataset_60.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # We want to stratify by task_type to ensure balanced sets.
    # We have 20 Fact, 20 Pattern, 20 NIAH.
    # Discovery (40) = ~13 per task. Confirmation (20) = ~7 per task.
    # For exactly 40/20, we can do 14/6, 13/7, 13/7. Let's just do random sample.
    
    fact = [d for d in data if d["task_type"] == "fact_recall"]
    pattern = [d for d in data if d["task_type"] == "pattern_induction"]
    niah = [d for d in data if d["task_type"] == "niah"]
    
    random.seed(42) # Reproducibility
    random.shuffle(fact)
    random.shuffle(pattern)
    random.shuffle(niah)
    
    discovery = fact[:13] + pattern[:14] + niah[:13]
    confirmation = fact[13:] + pattern[14:] + niah[13:]
    
    with open("dataset_discovery_40.json", "w", encoding="utf-8") as f:
        json.dump(discovery, f, indent=2)
        
    with open("dataset_confirmation_20.json", "w", encoding="utf-8") as f:
        json.dump(confirmation, f, indent=2)
        
    print(f"Created Discovery Set (N={len(discovery)}) and Confirmation Set (N={len(confirmation)})")

if __name__ == "__main__":
    main()
