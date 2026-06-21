# Syntax Used — Every Python Syntax Explained

This file explains every Python syntax and library call used in this project.
If you know basic Python (variables, loops, functions), this will fill in the gaps.

---

## Python Basics Used

### f-strings
```python
name = "Alice"
print(f"Hello {name}")  # prints: Hello Alice
```
The `f` before the string lets you put variables inside `{}`.

### Type Hints
```python
def add(a: int, b: int) -> int:
    return a + b
```
The `: int` says "this should be an integer". The `-> int` says "this returns an integer". Python doesn't enforce these — they're just documentation.

### Dictionaries
```python
d = {"key": "value", "age": 25}
d["key"]      # → "value"
d.get("missing", 0)  # → 0 (default if key missing)
```

### List Comprehensions
```python
squares = [x * x for x in range(10)]  # [0, 1, 4, 9, 16, ...]
```
A compact way to build a list. Read it as: "x*x for each x from 0 to 9".

### Tuple Unpacking
```python
point = (3, 4)
x, y = point  # x=3, y=4
```

### `with` Statement
```python
with open("file.txt") as f:
    data = f.read()
# file automatically closes when the block ends
```

### `os.makedirs(path, exist_ok=True)`
```python
import os
os.makedirs("outputs/phase0", exist_ok=True)
```
Creates the directory (and any parent directories). `exist_ok=True` means don't crash if it already exists.

---

## PyTorch Syntax

### Tensors
```python
import torch
t = torch.tensor([1.0, 2.0, 3.0])  # a 1D tensor (like a list of numbers on GPU)
t.shape  # → torch.Size([3])
```

### `torch.no_grad()`
```python
with torch.no_grad():
    output = model(input)
```
Tells PyTorch: "don't track gradients here". We're not training — just running the model. Saves memory.

### `.to(device)`
```python
model = model.to("cuda")  # move model to GPU
```
Moves a tensor or model to the specified device (CPU or GPU).

### `.half()`
```python
model = model.half()  # convert to float16 (half precision)
```
Uses 16-bit floats instead of 32-bit. Halves memory usage. Slightly less precise but fine for inference.

### `.squeeze()` / `.unsqueeze()`
```python
t = torch.tensor([[1, 2, 3]])  # shape (1, 3)
t.squeeze()    # shape (3) — removes dimensions of size 1
t.unsqueeze(0) # shape (1, 1, 3) — adds a dimension of size 1
```

### `.softmax(dim=-1)`
```python
scores = torch.tensor([1.0, 2.0, 3.0])
probs = scores.softmax(dim=-1)  # → [0.09, 0.24, 0.67]
```
Converts raw scores to probabilities (sum to 1). `dim=-1` means along the last dimension.

### `.topk(k)`
```python
values, indices = torch.tensor([5, 1, 8, 3]).topk(2)
# values = [8, 5], indices = [2, 0]
```
Returns the k largest values and their positions.

### `.mean()`, `.std()`
```python
t = torch.tensor([1.0, 2.0, 3.0])
t.mean()  # → 2.0
t.std()   # → 1.0
```

---

## HuggingFace Transformers

### Loading a Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```
Downloads the model and tokenizer from HuggingFace hub. Cached locally after first download.

### Tokenizing Text
```python
tokens = tokenizer("Hello world", return_tensors="pt")
# tokens["input_ids"] → tensor([[15496, 995]])
```
Converts text to numbers (token IDs). `return_tensors="pt"` gives PyTorch tensors.

### Running the Model
```python
output = model(**tokens, output_attentions=True)
```
`**tokens` unpacks the dict into keyword arguments. `output_attentions=True` tells the model to return attention weights (normally discarded to save memory).

### Attention Output
```python
attentions = output.attentions  # tuple of tensors, one per layer
attentions[0].shape  # → (batch, num_heads, seq_len, seq_len)
```
`attentions[layer][batch, head, query_token, key_token]` = how much `query_token` attended to `key_token` in that head/layer.

### `past_key_values`
```python
output = model(**tokens, use_cache=True)
past = output.past_key_values  # tuple of (K, V) per layer
past[0][0].shape  # → (batch, num_heads, seq_len, head_dim) — keys for layer 0
past[0][1].shape  # → same — values for layer 0
```
This IS the KV cache. Each layer stores its keys and values so they don't need recomputing.

---

## HuggingFace Datasets

### Loading WikiText-103
```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
text = dataset[0]["text"]  # first document's text
```

---

## NumPy

### Arrays
```python
import numpy as np
arr = np.array([1, 2, 3])
arr.mean()  # → 2.0
```

### `.argsort()`
```python
np.array([30, 10, 20]).argsort()  # → [1, 2, 0] (indices that would sort the array)
```

---

## Scikit-learn

### K-Means Clustering
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data)              # find 4 clusters in the data
labels = kmeans.labels_       # which cluster each point belongs to
centers = kmeans.cluster_centers_  # the center of each cluster
```

### Cosine Similarity
```python
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(A, B)  # shape (n_A, n_B), values -1 to 1
```
1 = identical direction, 0 = perpendicular, -1 = opposite.

---

## Matplotlib

### Basic Plot
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("My Plot")
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Heatmap
```python
plt.imshow(matrix, cmap="viridis", aspect="auto")
plt.colorbar()
```
Displays a 2D matrix as colors. `cmap` = color scheme. `viridis` = yellow-green-blue.

---

## Pickle (Saving Python Objects)

```python
import pickle

# Save
with open("data.pkl", "wb") as f:   # wb = write bytes
    pickle.dump(my_object, f)

# Load
with open("data.pkl", "rb") as f:   # rb = read bytes
    my_object = pickle.load(f)
```
Saves any Python object (dicts, lists, numpy arrays) to a file. `wb`/`rb` = binary mode.

---

## `tqdm` (Progress Bars)

```python
from tqdm import tqdm
for item in tqdm(my_list, desc="Processing"):
    do_something(item)
# Shows: Processing: 45%|████▌     | 45/100
```

---

## `argparse` (Command Line Arguments)

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=256)
args = parser.parse_args()
print(args.budget)  # → 256 (or whatever the user passed)
```
Run with: `python script.py --budget 512`

---

## `torch.cuda` (GPU Memory)

```python
torch.cuda.memory_allocated()    # bytes currently used
torch.cuda.max_memory_allocated() # peak bytes used
torch.cuda.empty_cache()         # free unused cached memory
```
