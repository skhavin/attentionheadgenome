# Generate text using proactive KV cache eviction.
# Encodes a document, builds retention mask, prunes KV cache, then generates.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from phase2.predict_prototypes import predict_prototypes
from phase2.build_retention_mask import predict_retention_mask
from phase3.kv_cache_wrapper import apply_retention_mask
from config import (MODEL_NAME, DEVICE, USE_FP16, PROTOTYPES_PATH,
                    MAX_SEQ_LEN, GENERATE_LENGTH)

def generate_with_proactive_eviction(model, tokenizer, text, prototypes, budget=256):
    """Full pipeline: encode → predict prototype → build mask → prune KV → generate."""

    # Step 1: Tokenize the input document
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"].to(DEVICE)

    # Step 2: Run the prefill pass (process the whole document)
    with torch.no_grad():
        output = model(input_ids, use_cache=True)
    past_kv = output.past_key_values

    # Step 3: Predict prototypes and build retention mask
    predictions = predict_prototypes(None, prototypes)  # simple version
    masks = predict_retention_mask(prototypes, predictions, input_ids.shape[1], budget)

    # Step 4: Prune the KV cache
    pruned_kv = apply_retention_mask(past_kv, masks, budget, device=DEVICE)

    # Step 5: Generate tokens one by one using the pruned cache
    generated = []
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())

    for _ in range(GENERATE_LENGTH - 1):
        with torch.no_grad():
            output = model(next_token, past_key_values=pruned_kv, use_cache=True)
        pruned_kv = output.past_key_values  # new cache includes the new token
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

    return tokenizer.decode(generated)

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)

    # Use validation split
    articles = load_articles(split="validation", max_articles=5)

    for i, text in enumerate(articles):
        print(f"\n--- Article {i} (first 100 chars): {text[:100]}...")
        result = generate_with_proactive_eviction(model, tokenizer, text, prototypes, budget=256)
        print(f"Generated: {result}")

if __name__ == "__main__":
    main()
