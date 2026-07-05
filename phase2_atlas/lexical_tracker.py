import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["unsloth/Llama-3.2-1B"]

print("Loading dataset...")
with open("outputs/phase2_atlas/dataset.json") as f:
    dataset = json.load(f)
# Grab a chunk of text
text = " ".join([sample["text"] for sample in dataset["wikitext"][:5]])

for MODEL_NAME in MODELS:
    SAFE_MODEL = MODEL_NAME.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"MODEL: {SAFE_MODEL}")
    print(f"{'='*60}")
    
    atlas_path = f"outputs/phase2_atlas/{SAFE_MODEL}_head_atlas.json"
    if not os.path.exists(atlas_path):
        continue
        
    with open(atlas_path) as f:
        atlas = json.load(f)
        
    # Group heads by class
    class_heads = {"Induction": [], "Local": [], "Retrieval": [], "Sink": []}
    for k, h in atlas["heads"].items():
        c = h.get("class_label")
        if c in class_heads:
            class_heads[c].append((h["layer"], h["head"]))
            
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    model.eval()
    
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = ids["input_ids"][0].tolist()
    tokens = [tok.decode([tid]).replace('\n', '\\n') for tid in input_ids]
    
    with torch.no_grad():
        out = model(**ids, output_attentions=True)
        
    class_words = {c: Counter() for c in class_heads}
    
    # Analyze attentions
    # attentions shape: tuple of layers, each (batch, num_heads, seq, seq)
    for c, heads in class_heads.items():
        if not heads: continue
        
        for (L, H) in heads:
            attn = out.attentions[L][0, H] # (seq, seq)
            
            # Find tokens that received > 30% of attention mass from any query
            # We look at the dimension 1 (attended-to tokens)
            max_attn_received, _ = attn.max(dim=0)
            
            for i, val in enumerate(max_attn_received.tolist()):
                if val > 0.3:
                    class_words[c][tokens[i].strip()] += 1
                    
    # Print top words for each class
    for c, counts in class_words.items():
        if not counts: continue
        # Filter out empty strings or basic punctuation if desired, but let's see everything
        top = [w for w, count in counts.most_common(10) if w.strip()]
        print(f"Top 10 heavily-attended tokens for {c} heads:")
        print(f"  {top}")
