import os
import sys
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase7.moe.moe_patcher import MoEPatcher

def evaluate_ppl(model, tokenizer, device, num_chunks=40):
    model.eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    
    max_length = 1024
    stride = 512
    seq_len_total = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    num_evals = min(num_chunks, (seq_len_total - 1) // stride + 1)
    
    for begin_loc in tqdm(range(0, seq_len_total, stride), total=num_evals, desc="Evaluating PPL"):
        if len(nlls) >= num_evals:
            break
        end_loc = min(begin_loc + max_length, seq_len_total)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len_total:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    
    print("Installing MoEPatcher...")
    patcher = MoEPatcher(model)
    
    # Force all routers to always output p_full = 1.0
    # We can do this by patching the router forward or LayerRouter forward
    for r in patcher.routers.values():
        def custom_forward(features, hard_routing=False, r=r):
            # return shape [B, H, 4] with 1.0 in index 3 (full)
            B, H, _ = features.shape
            probs = torch.zeros(B, H, 4, device=features.device, dtype=features.dtype)
            probs[:, :, 3] = 1.0
            r._last_probs = probs.detach()
            return probs
        r.forward = custom_forward

    print("Evaluating PPL with always-full MoEPatcher...")
    ppl = evaluate_ppl(model, tokenizer, device, num_chunks=40)
    print(f"PPL: {ppl:.4f}")

if __name__ == "__main__":
    main()
