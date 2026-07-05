import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

# 50 prompt pairs across 5 categories (10 each)
PROMPT_PAIRS = [
    # -- Geography (10) --
    ("The capital of France is Paris.", "The weather today is sunny and warm.", " The capital of France is"),
    ("The Nile is the longest river in the world.", "She enjoyed reading books in the evening.", " The Nile is the longest"),
    ("The Eiffel Tower is located in Paris.", "The stock market closed higher yesterday.", " The Eiffel Tower is located in"),
    ("Mount Everest is the tallest mountain on Earth.", "He enjoyed cooking pasta for dinner.", " Mount Everest is the tallest"),
    ("The Pacific Ocean is the largest ocean on Earth.", "He forgot to bring his umbrella to work.", " The Pacific Ocean is the"),
    ("The Amazon River flows through Brazil.", "She spent the afternoon painting in her studio.", " The Amazon River flows through"),
    ("Rome is the capital of Italy.", "She planted roses in her garden this spring.", " Rome is the capital of"),
    ("The Sahara is the largest hot desert in the world.", "The train arrived at the station late.", " The Sahara is the largest"),
    ("Japan is an island nation in East Asia.", "The children played outside all afternoon.", " Japan is an island nation in"),
    ("The Great Wall of China stretches over 13000 miles.", "The conference was held in a large hall.", " The Great Wall of China stretches over"),
    # -- Science (10) --
    ("The speed of light is 299792458 meters per second.", "The dog ran quickly across the field.", " The speed of light is"),
    ("Water boils at 100 degrees Celsius at sea level.", "The train arrived at the station late.", " Water boils at"),
    ("The chemical formula for water is H2O.", "The children played outside all afternoon.", " The chemical formula for water is"),
    ("Oxygen has the atomic number 8.", "The library closed early due to renovations.", " Oxygen has the atomic number"),
    ("DNA stands for deoxyribonucleic acid.", "The children built a sandcastle on the beach.", " DNA stands for"),
    ("Photosynthesis converts sunlight into chemical energy.", "They watched the fireworks from the hilltop.", " Photosynthesis converts sunlight into"),
    ("The human body has 206 bones.", "The cat curled up on the warm windowsill.", " The human body has"),
    ("The Earth orbits the Sun once every 365 days.", "He baked a cake for the birthday party.", " The Earth orbits the Sun once every"),
    ("Gravity was described by Newton as an attractive force between masses.", "She reorganized the books on the shelf.", " Gravity was described by Newton as"),
    ("Light travels faster than sound in air.", "They adopted a puppy from the shelter.", " Light travels faster than"),
    # -- History (10) --
    ("The Berlin Wall fell in 1989.", "He enjoyed hiking in the national park.", " The Berlin Wall fell in"),
    ("Shakespeare was born in Stratford-upon-Avon.", "The supermarket was crowded on Saturday morning.", " Shakespeare was born in"),
    ("The French Revolution began in 1789.", "She organized her wardrobe on Sunday afternoon.", " The French Revolution began in"),
    ("World War II ended in 1945.", "He went jogging along the river each morning.", " World War II ended in"),
    ("The Roman Empire fell in 476 AD.", "She prepared a fresh salad for dinner.", " The Roman Empire fell in"),
    ("Christopher Columbus arrived in the Americas in 1492.", "The dog learned a new trick in the park.", " Christopher Columbus arrived in the Americas in"),
    ("The Declaration of Independence was signed in 1776.", "He repaired the bicycle in the garage.", " The Declaration of Independence was signed in"),
    ("The Moon landing happened in July 1969.", "She watered the plants on the balcony.", " The Moon landing happened in"),
    ("Napoleon was exiled to the island of Elba.", "The library had a new collection of novels.", " Napoleon was exiled to"),
    ("The Treaty of Versailles was signed in 1919.", "The children decorated the classroom with drawings.", " The Treaty of Versailles was signed in"),
    # -- Literature (10) --
    ("Leonardo da Vinci painted the Mona Lisa.", "The new restaurant opened last week downtown.", " Leonardo da Vinci painted the"),
    ("Hamlet is a play written by Shakespeare.", "The supermarket had a sale on fresh produce.", " Hamlet is a play written by"),
    ("George Orwell wrote the novel 1984.", "She arranged fresh flowers in a vase.", " George Orwell wrote the novel"),
    ("The Great Gatsby was written by F. Scott Fitzgerald.", "He mended the fence in the backyard.", " The Great Gatsby was written by"),
    ("Don Quixote was written by Miguel de Cervantes.", "The children built a fort out of pillows.", " Don Quixote was written by"),
    ("Homer wrote the Iliad and the Odyssey.", "The museum displayed a new art collection.", " Homer wrote the"),
    ("Tolstoy wrote War and Peace.", "The bakery sold fresh bread every morning.", " Tolstoy wrote"),
    ("Mary Shelley wrote Frankenstein.", "The park was crowded with families on Sunday.", " Mary Shelley wrote"),
    ("Charles Dickens wrote A Tale of Two Cities.", "She fixed the leaking pipe in the kitchen.", " Charles Dickens wrote"),
    ("Dante wrote the Divine Comedy.", "They rearranged the furniture in the living room.", " Dante wrote"),
    # -- Mathematics (10) --
    ("The Pythagorean theorem states that a squared plus b squared equals c squared.", "She read a mystery novel before going to sleep.", " The Pythagorean theorem states"),
    ("Pi is approximately equal to 3.14159.", "He took the dog for a walk in the park.", " Pi is approximately equal to"),
    ("The square root of 144 is 12.", "She watered the plants and trimmed the hedges.", " The square root of 144 is"),
    ("Euler's number e is approximately 2.71828.", "He painted the fence white over the weekend.", " Euler's number e is approximately"),
    ("A prime number has no divisors other than 1 and itself.", "The children played board games on rainy days.", " A prime number has no divisors other than"),
    ("The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5.", "She made lemonade for the summer fair.", " The Fibonacci sequence starts with"),
    ("The sum of angles in a triangle is 180 degrees.", "He organized his stamp collection by country.", " The sum of angles in a triangle is"),
    ("A circle's circumference equals pi times the diameter.", "She learned to knit from her grandmother.", " A circle's circumference equals"),
    ("The derivative of x squared is 2x.", "He assembled a model airplane kit.", " The derivative of x squared is"),
    ("The quadratic formula solves ax squared plus bx plus c equals zero.", "She attended a pottery class on Saturdays.", " The quadratic formula solves"),
]

def attention_entropy(attn_weights):
    last_pos = attn_weights[:, -1, :].float()
    nan_mask = torch.isnan(last_pos)
    last_pos = torch.where(nan_mask, torch.zeros_like(last_pos), last_pos)
    row_sum = last_pos.sum(dim=-1, keepdim=True)
    zero_rows = (row_sum == 0).squeeze(-1)
    if zero_rows.any():
        last_pos[zero_rows] = 1.0 / last_pos.shape[-1]
        row_sum = last_pos.sum(dim=-1, keepdim=True)
    p = last_pos / row_sum + 1e-12
    p = p / p.sum(dim=-1, keepdim=True)
    return (-torch.sum(p * torch.log(p), dim=-1)).cpu().numpy()

def run_prompt(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    return [attention_entropy(a[0]) for a in out.attentions]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium", attn_implementation="eager").to(device)
    model.eval()

    L, H = 24, 16
    
    # Track delta for each pair, for reference heads
    ref_heads = {
        "Sink": (5, 11),
        "Local": (23, 5),
        "Retrieval": (15, 8),
        "Induction": (9, 3)
    }
    
    head_deltas = {name: [] for name in ref_heads}

    print("Running stability test on 50 prompt pairs...")
    for i, (ctx_m, ctx_nm, query) in enumerate(PROMPT_PAIRS):
        me = run_prompt(model, tok, ctx_m + query, device)
        nme = run_prompt(model, tok, ctx_nm + query, device)
        
        for name, (l, h) in ref_heads.items():
            match_ent = me[l][h]
            nonmatch_ent = nme[l][h]
            delta = float(nonmatch_ent - match_ent)
            head_deltas[name].append(delta)

    print("\n--- Entropy Collapse Stability Results ---")
    for name, deltas in head_deltas.items():
        arr = np.array(deltas)
        mean = arr.mean()
        std = arr.std()
        print(f"[{name} Head L{ref_heads[name][0]}H{ref_heads[name][1]}]")
        print(f"  Delta Mean: {mean:.4f}")
        print(f"  Delta StdDev: {std:.4f}")
        print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}\n")

if __name__ == "__main__":
    main()
