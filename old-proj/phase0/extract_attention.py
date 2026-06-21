# Run GPT-2 on 10 sentences and save the raw attention tensors.
# Output: one .pt file per sentence in outputs/phase0/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, DEVICE, USE_FP16, PHASE0_DIR, PHASE0_SENTENCES

def main():
    os.makedirs(PHASE0_DIR, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # gpt2 tokenizer works for gpt2-medium
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")  # use gpt2-small for phase0 (faster)
    model.eval()
    model.to(DEVICE)
    if USE_FP16:
        model.half()

    for i, sentence in enumerate(PHASE0_SENTENCES):
        # Tokenize
        tokens = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        # Run model, get attention weights
        with torch.no_grad():
            output = model(**tokens, output_attentions=True)

        # attentions is a tuple of (batch, heads, seq, seq) — one per layer
        # Move to CPU and save
        attentions = tuple(a.cpu().float() for a in output.attentions)
        save_path = os.path.join(PHASE0_DIR, f"attention_{i}.pt")
        torch.save({"sentence": sentence, "attentions": attentions}, save_path)
        print(f"[{i+1}/10] Saved {save_path}")

    print("Done! Attention tensors saved.")

if __name__ == "__main__":
    main()
