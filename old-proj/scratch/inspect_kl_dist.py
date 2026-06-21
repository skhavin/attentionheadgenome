import json
import numpy as np

with open("outputs/phase7/head_audit.json") as f:
    data = json.load(f)

kl_nats = [x["kl_natural_max"] for x in data]
kl_copies = [x["kl_copy_max"] for x in data]

print("KL Natural Max Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {np.percentile(kl_nats, p):.6f}")

print("\nKL Copy Max Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {np.percentile(kl_copies, p):.6f}")

print("\nNumber of heads with KL Natural < 0.01:", sum(1 for x in kl_nats if x < 0.01))
print("Number of heads with KL Natural < 0.05:", sum(1 for x in kl_nats if x < 0.05))
print("Number of heads with KL Natural < 0.10:", sum(1 for x in kl_nats if x < 0.10))
print("Number of heads with KL Natural < 0.50:", sum(1 for x in kl_nats if x < 0.50))
print("Number of heads with KL Natural < 1.00:", sum(1 for x in kl_nats if x < 1.00))
