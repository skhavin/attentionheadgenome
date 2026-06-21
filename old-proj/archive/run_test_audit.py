import torch
import json
import pickle
import sys
sys.path.append("phase7")
from audit_heads import load_model, load_dataset_chunks, run_audit

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = load_model(model_name, device)
    
    print("Loading tiny dataset chunks...")
    chunks_nat = load_dataset_chunks(tokenizer, dataset_name="wikitext", seq_len=1024, num_docs=2)
    chunks_copy = load_dataset_chunks(tokenizer, dataset_name="wikitext", seq_len=1024, num_docs=1)
    chunks_niah = load_dataset_chunks(tokenizer, dataset_name="wikitext", seq_len=1024, num_docs=1)
    
    print("Running audit...")
    audit_data = run_audit(model, chunks_nat, chunks_copy, chunks_niah, device,
                           num_sink_tokens=4, local_window=64)
    
    with open("test_audit.json", "w") as f:
        json.dump(audit_data, f, indent=2)

if __name__ == "__main__":
    main()
