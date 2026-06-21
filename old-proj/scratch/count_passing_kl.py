import json

with open("outputs/phase7/head_audit.json") as f:
    data = json.load(f)

print("Total entries:", len(data))
sinks = [x for x in data if x["type"] == "sink"]
locals = [x for x in data if x["type"] == "local"]

kl_thresh = 0.01

sinks_pass = [x for x in sinks if x["kl_natural_max"] < kl_thresh and x["kl_copy_max"] < kl_thresh]
locals_pass = [x for x in locals if x["kl_natural_max"] < kl_thresh and x["kl_copy_max"] < kl_thresh]

print(f"Sinks passing KL threshold (< {kl_thresh}): {len(sinks_pass)} / {len(sinks)}")
print(f"Locals passing KL threshold (< {kl_thresh}): {len(locals_pass)} / {len(locals)}")

print("\nSample passing Sinks:")
for x in sinks_pass[:10]:
    print(f"L{x['layer']}H{x['head']}: out_linf_nat_max={x['out_l_inf_rel_natural_max']:.5f}, kl_nat_max={x['kl_natural_max']:.5f}, kl_copy_max={x['kl_copy_max']:.5f}")

print("\nSample passing Locals:")
for x in locals_pass[:10]:
    print(f"L{x['layer']}H{x['head']}: out_linf_nat_max={x['out_l_inf_rel_natural_max']:.5f}, kl_nat_max={x['kl_natural_max']:.5f}, kl_copy_max={x['kl_copy_max']:.5f}")
