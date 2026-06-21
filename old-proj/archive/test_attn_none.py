import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('gpt2-medium', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')

import random
rng = random.Random(42)
vocab = list(tokenizer.get_vocab().keys())
text = "The quick brown fox"
ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]

seq_len = 512
extra = " ".join(rng.choice(vocab) for _ in range(seq_len * 2))
text = text + " " + extra
ids  = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
ids = ids[:, :seq_len].to(model.device)

with torch.no_grad():
    out = model(ids, output_attentions=True, use_cache=False)

print("attentions is None?", out.attentions is None)
if out.attentions is not None:
    print("attentions[0] is None?", out.attentions[0] is None)
