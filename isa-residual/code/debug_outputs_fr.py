import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    discovery = get_prompts("../../isa-head/dataset_discovery_336.json")
    fr_prompts = [p for p in discovery if p["task_type"] == "fact_recall"][:5]
    
    for p in fr_prompts:
        prompt_text = p["prompt"]
        target_text = p["target"]
        tokens = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**tokens)
            
        logits = outputs.logits[0, -1, :].float()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, 5)
        print(f"\nPrompt: {prompt_text}")
        print(f"Target: '{target_text}' (stripped lower: '{target_text.strip().lower()}')")
        for prob, idx in zip(top_probs, top_indices):
            token_str = tokenizer.decode(idx)
            print(f"  Token: '{token_str}' (prob: {prob.item():.4f})")

if __name__ == "__main__":
    main()
