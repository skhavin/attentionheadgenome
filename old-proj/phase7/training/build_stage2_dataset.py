import os
import sys
import json
import random
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def build_copy_trigger_prompts_simple(tokenizer, seq_len, num_prompts):
    vocab = ["cat", "dog", "mat", "sat", "hat", "bat", "rat", "tree",
             "book", "cook", "look", "hook", "took", "good", "wood"]
    rng = random.Random(42)
    prompts = []
    for _ in range(num_prompts):
        n = rng.randint(1, 5)
        ngram = [rng.choice(vocab) for _ in range(n)]
        dist = rng.choice([5, 20, 100])
        filler = [rng.choice(vocab) for _ in range(dist)]
        text = " ".join(ngram + filler + ngram)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        prompts.append(ids.tolist())
    return prompts

def build_natural_prompts_simple(tokenizer, seq_len, num_docs):
    from datasets import load_dataset
    from config import DATASET_NAME, DATASET_CONFIG
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    
    # Just take chunks of tokens
    full_text = " ".join(row["text"] for row in ds.select(range(5000)) if row["text"].strip())
    all_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i: i + seq_len].tolist())
        if len(chunks) >= num_docs:
            break
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--natural_docs", type=int, default=10)
    parser.add_argument("--copy_trigger_docs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output", default="outputs/phase7/stage2_mixed.jsonl")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating {args.natural_docs} natural docs...")
    natural = build_natural_prompts_simple(tokenizer, args.seq_len, args.natural_docs)
    
    print(f"Generating {args.copy_trigger_docs} copy-trigger docs...")
    copy_triggers = build_copy_trigger_prompts_simple(tokenizer, args.seq_len, args.copy_trigger_docs)

    data = [{"tokens": t, "type": "natural"} for t in natural] + \
           [{"tokens": t, "type": "copy"} for t in copy_triggers]
    
    random.seed(42)
    random.shuffle(data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(data)} documents to {args.output}")

if __name__ == "__main__":
    main()
