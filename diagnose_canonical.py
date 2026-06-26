import json
from collections import Counter

# Look at the actual structure of robust entropy heads
d = json.load(open('outputs/phase1/robust_entropy_gpt2.json'))
heads = d.get('heads', {})
print('Sample head keys:', list(heads.keys())[:3])
print('Sample head value:', str(list(heads.values())[0])[:400])
print()

# Also check llama
d2 = json.load(open('outputs/phase1/llama1b_retrieval_entropy.json'))
heads2 = d2.get('heads', {})
print('Llama sample head keys:', list(heads2.keys())[:3])
print('Llama sample head value:', str(list(heads2.values())[0])[:400])
