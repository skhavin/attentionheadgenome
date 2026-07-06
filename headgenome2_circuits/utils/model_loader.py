import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
    "llama-1b": "meta-llama/Llama-3.2-1B",
    "gemma-2b": "google/gemma-2b",
    "gpt2": "gpt2"
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model_and_tokenizer(model_key="qwen-0.5b", output_attentions=True, output_hidden_states=True):
    """
    Loads the requested model and tokenizer.
    Ensures standard formatting and padding tokens are set.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODELS.keys())}")
        
    model_id = MODELS[model_key]
    print(f"Loading {model_id} on {get_device()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if get_device() != "cpu" else torch.float32,
        device_map=get_device(),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states
    )
    
    model.eval()
    return model, tokenizer
