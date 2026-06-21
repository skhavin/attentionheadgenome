from transformers import AutoConfig
try:
    print(AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct"))
except Exception as e:
    print("GATED_ERROR:", e)
