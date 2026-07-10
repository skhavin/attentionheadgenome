import json
import numpy as np

with open('phase1_results.json', 'r') as f:
    data = json.load(f)

raw = data['raw']
n = len(raw)

reals = [r['real_crossover_layer'] for r in raw]
shuffles = [r['shuffled_crossover_layer'] for r in raw]

ceiling_hits = sum(1 for s in shuffles if s == 27)
ceiling_frac = ceiling_hits / n

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    gt = sum(1 for i in x for j in y if i > j)
    lt = sum(1 for i in x for j in y if i < j)
    return (gt - lt) / (n1 * n2)

delta = cliffs_delta(reals, shuffles)

print(f"N = {n}")
print(f"Ceiling Hits = {ceiling_hits} ({ceiling_frac*100:.1f}%)")
print(f"Cliff's Delta = {delta:.4f}")
