# check_quant_histograms.py
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "outputs", "phase1", "dataset_index.json")

sys.path.insert(0, os.path.join(ROOT, "phase1"))
from step3_profile_shared import load_articles_from_index, extract_patterns_one_doc

def load_model(quantize):
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quantize:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager"
        )
    model.eval()
    return model, tokenizer

def main():
    texts = load_articles_from_index(INDEX_PATH, num_docs=5)
    
    print("Profiling BF16...")
    m_bf16, tok = load_model(False)
    p_bf16 = [extract_patterns_one_doc(m_bf16, tok, t) for t in texts]
    p_bf16 = [p for p in p_bf16 if p is not None]
    del m_bf16
    torch.cuda.empty_cache()
    
    print("Profiling 4-bit...")
    m_4bit, _ = load_model(True)
    p_4bit = [extract_patterns_one_doc(m_4bit, tok, t) for t in texts]
    p_4bit = [p for p in p_4bit if p is not None]
    del m_4bit
    torch.cuda.empty_cache()
    
    # Calculate cosine similarity of mean histograms
    keys = sorted(p_bf16[0].keys())
    similarities = []
    
    for layer, head in keys:
        hists_bf16 = np.mean([doc[(layer, head)] for doc in p_bf16], axis=0)
        hists_4bit = np.mean([doc[(layer, head)] for doc in p_4bit], axis=0)
        
        # Cosine similarity
        num = np.dot(hists_bf16, hists_4bit)
        den = np.linalg.norm(hists_bf16) * np.linalg.norm(hists_4bit)
        cos_sim = num / den if den > 0 else 0.0
        similarities.append(cos_sim)
        
    print(f"\nMean Cosine Similarity: {np.mean(similarities):.6f}")
    print(f"Min Cosine Similarity: {np.min(similarities):.6f}")
    print(f"Max Cosine Similarity: {np.max(similarities):.6f}")
    print(f"Fraction > 0.95: {np.mean(np.array(similarities) > 0.95):.4f}")
    print(f"Fraction > 0.99: {np.mean(np.array(similarities) > 0.99):.4f}")

if __name__ == "__main__":
    main()
