# System Architecture

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (done once)                          │
│                                                                     │
│  WikiText-103 docs                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │ GPT-2 Medium │ ──▶ │ Attention Patterns│ ──▶ │  K-Means       │  │
│  │ (frozen)     │     │ per (layer, head) │     │  Clustering    │  │
│  └──────────────┘     └──────────────────┘     └───────┬────────┘  │
│                                                         │           │
│                                                         ▼           │
│                                                 ┌──────────────┐   │
│                                                 │ prototypes   │   │
│                                                 │ .pkl         │   │
│                                                 └──────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE (per new document)                       │
│                                                                     │
│  New document                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │ Lightweight   │ ──▶ │ Nearest-centroid │ ──▶ │ Retention      │  │
│  │ Encoder       │     │ matching         │     │ Mask           │  │
│  │ (embed only)  │     │ (cosine sim)     │     │ (binary)       │  │
│  └──────────────┘     └──────────────────┘     └───────┬────────┘  │
│                                                         │           │
│                                                         ▼           │
│                                                 ┌──────────────┐   │
│                                                 │ Pruned KV    │   │
│                                                 │ Cache        │   │
│                                                 └──────┬───────┘   │
│                                                        │            │
│  User query ──────────────────────────────────────────▶│            │
│                                                        ▼            │
│                                                 ┌──────────────┐   │
│                                                 │ Fast         │   │
│                                                 │ Attention    │   │
│                                                 │ (budget only)│   │
│                                                 └──────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Attention Profiler (`phase1/run_profiling.py`)
- Runs frozen GPT-2 Medium on WikiText-103 documents
- Extracts attention weights using `output_attentions=True`
- For each (layer, head), records relative positions of top-k attended tokens
- Output: numpy arrays of attention patterns

### 2. Prototype Builder (`phase1/build_prototypes.py`)
- Takes attention patterns from profiler
- Runs K-Means clustering per (layer, head)
- Each cluster centroid = one behavioral prototype
- Output: `prototypes.pkl` — dict mapping `(layer, head)` → list of centroids

### 3. Prototype Predictor (`phase2/predict_prototypes.py`)
- Takes a new document (raw text)
- Embeds it using GPT-2's embedding table (no forward pass)
- Computes cosine similarity to each prototype centroid
- Output: predicted prototype index per (layer, head)

### 4. Retention Mask Builder (`phase2/build_retention_mask.py`)
- Takes predicted prototypes + document tokens
- Converts each prototype into a binary mask over token positions
- Output: dict mapping `(layer, head)` → boolean mask of length n_tokens

### 5. KV Cache Wrapper (`phase3/kv_cache_wrapper.py`)
- Wraps HuggingFace's `past_key_values` (tuple of (K, V) tensors)
- Before each generation step, applies retention mask to slice out pruned tokens
- Transparent to the model — it just sees smaller K, V tensors

## Tensor Shapes

```
GPT-2 Medium:
  - Layers: 24
  - Heads per layer: 16
  - Head dimension: 64
  - Total hidden: 1024

Attention output shape: (batch, num_heads, seq_len, seq_len)
KV cache shape per layer: 2 × (batch, num_heads, seq_len, head_dim)

At seq_len=1024 in fp16:
  KV cache per layer = 2 × 1 × 16 × 1024 × 64 × 2 bytes = 4 MB
  Total (24 layers) = 96 MB

At seq_len=10000:
  Total = ~940 MB  ← this is what we're trying to shrink
```

## File Dependency Graph

```
config.py ◄──── everything imports this
    │
    ▼
phase0/ ──── standalone, just visualization
    │
    ▼
phase1/ ──── produces prototypes.pkl
    │
    ▼
phase2/ ──── uses prototypes.pkl, produces retention masks
    │
    ▼
phase3/ ──── uses retention masks, produces benchmark results
    │
    ▼
phase4/ ──── repeats phase1 on different models (Qwen, LLaMA)
    │
    ▼
phase5/ ──── evaluates long-context benchmarks on PG-19 dataset
```
