# Compute simple statistics to identify head behaviors:
# - Locality score: does this head mostly attend to nearby tokens?
# - Sink score: does this head mostly attend to the first token?
# Output: printed table of scores per layer/head.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from config import PHASE0_DIR

def main():
    # Load all 10 sentences' attention data
    all_locality = {}   # (layer, head) -> list of scores
    all_sink = {}

    for i in range(10):
        path = os.path.join(PHASE0_DIR, f"attention_{i}.pt")
        if not os.path.exists(path):
            print(f"Skipping {path} (not found)")
            continue
        data = torch.load(path, weights_only=False)
        attentions = data["attentions"]

        for layer_idx, layer_attn in enumerate(attentions):
            attn = layer_attn[0]  # (heads, seq, seq)
            num_heads, seq_len, _ = attn.shape

            for head_idx in range(num_heads):
                head = attn[head_idx].numpy()  # (seq, seq)
                key = (layer_idx, head_idx)

                # Locality: average attention to tokens within distance 3
                local_weight = 0.0
                for q in range(seq_len):
                    start = max(0, q - 3)
                    end = min(seq_len, q + 4)
                    local_weight += head[q, start:end].sum()
                locality = local_weight / seq_len
                all_locality.setdefault(key, []).append(locality)

                # Sink: average attention to token 0
                sink = head[:, 0].mean()
                all_sink.setdefault(key, []).append(sink)

    # Print top local and top sink heads
    avg_locality = {k: np.mean(v) for k, v in all_locality.items()}
    avg_sink = {k: np.mean(v) for k, v in all_sink.items()}

    print("\n=== Top 10 LOCAL heads (attend to nearby tokens) ===")
    for (l, h), score in sorted(avg_locality.items(), key=lambda x: -x[1])[:10]:
        print(f"  Layer {l:2d}, Head {h:2d}: locality = {score:.3f}")

    print("\n=== Top 10 SINK heads (attend to first token) ===")
    for (l, h), score in sorted(avg_sink.items(), key=lambda x: -x[1])[:10]:
        print(f"  Layer {l:2d}, Head {h:2d}: sink = {score:.3f}")

if __name__ == "__main__":
    main()
