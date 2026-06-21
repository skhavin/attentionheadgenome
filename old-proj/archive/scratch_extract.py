import pickle
import numpy as np

path = 'outputs/phase4/meta-llama-3.1-8b-bnb-4bit_attention_patterns.pkl'
print(f"Loading {path}...")
with open(path, 'rb') as f:
    docs = pickle.load(f)

keys = sorted(docs[0].keys())
num_layers = max(l for l, h in keys) + 1
num_heads = max(h for l, h in keys) + 1

mean_dist = np.zeros((num_layers, num_heads, 512))
for layer, head in keys:
    data = np.array([d[(layer, head)] for d in docs if (layer, head) in d])
    mean_dist[layer, head] = data.mean(axis=0)

cdf = np.cumsum(mean_dist, axis=-1)
window = np.argmax(cdf >= 0.95, axis=-1)

print("95th percentile radius (window size) per layer and head:")
for layer in range(num_layers):
    # Print comma separated for easy reading
    vals = ", ".join(f"{w:3d}" for w in window[layer])
    print(f"Layer {layer:02d}: [{vals}]")
