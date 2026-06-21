# Embed documents using GPT-2's embedding table only (no forward pass).
# Just look up token embeddings and average them. Super cheap.
# Output: doc_embeddings.pkl in outputs/phase2/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from config import MODEL_NAME, DEVICE, PHASE2_DIR, PREDICTION_TEST_DOCS, MAX_SEQ_LEN

def embed_document(text, tokenizer, embedding_layer):
    """Embed a document by averaging its token embeddings. No forward pass needed."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = tokens["input_ids"][0]

    with torch.no_grad():
        embeddings = embedding_layer(input_ids.to(DEVICE))  # (seq_len, hidden_dim)
        doc_embedding = embeddings.mean(dim=0).cpu().numpy()

    return doc_embedding

def main():
    os.makedirs(PHASE2_DIR, exist_ok=True)

    print("Loading model (embedding layer only)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    embedding_layer = model.transformer.wte

    # Use validation split for evaluation (never same data as profiling)
    articles = load_articles(split="validation", max_articles=PREDICTION_TEST_DOCS)

    print(f"Embedding {len(articles)} articles...")
    embeddings = []
    for text in tqdm(articles, desc="Embedding"):
        emb = embed_document(text, tokenizer, embedding_layer)
        embeddings.append(emb)

    save_path = os.path.join(PHASE2_DIR, "doc_embeddings.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"embeddings": np.array(embeddings), "texts": articles}, f)
    print(f"Saved {len(embeddings)} embeddings to {save_path}")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
