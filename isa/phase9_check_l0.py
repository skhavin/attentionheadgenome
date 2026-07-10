import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    with open("dataset_discovery_40.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)[:5]

    for item in prompts:
        prompt = item["prompt"]
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            
        # output_hidden_states[0] is the embedding layer (or L0 output)
        l0_hidden = outputs.hidden_states[1] # index 0 is embedding, 1 is after layer 0
        logits_l0 = model.lm_head(model.model.norm(l0_hidden[0, -1, :]))
        top_token_l0 = tokenizer.decode(logits_l0.argmax().item())
        
        last_input_token = tokenizer.decode(tokens.input_ids[0, -1].item())
        
        print(f"Prompt: {prompt}")
        print(f"Last Input Token: '{last_input_token}'")
        print(f"Layer 0 Prediction: '{top_token_l0}'")
        print("-" * 50)

if __name__ == "__main__":
    main()
