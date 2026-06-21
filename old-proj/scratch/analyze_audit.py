import json

with open("outputs/phase7/head_audit.json") as f:
    data = json.load(f)

print("Total entries:", len(data))
sinks = [x for x in data if x["type"] == "sink"]
locals = [x for x in data if x["type"] == "local"]

print("\nTop 10 Sinks by lowest out_l_inf_rel_natural_max:")
sinks_sorted = sorted(sinks, key=lambda x: x["out_l_inf_rel_natural_max"] if x["out_l_inf_rel_natural_max"] is not None else 999)
for x in sinks_sorted[:10]:
    print(f"L{x['layer']}H{x['head']}: out_linf_nat_max={x['out_l_inf_rel_natural_max']:.5f}, attn_linf_nat_max={x['attn_l_inf_natural_max']:.5f}, kl_nat_max={x['kl_natural_max']:.5f}, kl_copy_max={x['kl_copy_max']:.5f}")

print("\nTop 10 Locals by lowest out_l_inf_rel_natural_max:")
locals_sorted = sorted(locals, key=lambda x: x["out_l_inf_rel_natural_max"] if x["out_l_inf_rel_natural_max"] is not None else 999)
for x in locals_sorted[:10]:
    print(f"L{x['layer']}H{x['head']}: out_linf_nat_max={x['out_l_inf_rel_natural_max']:.5f}, attn_linf_nat_max={x['attn_l_inf_natural_max']:.5f}, kl_nat_max={x['kl_natural_max']:.5f}, kl_copy_max={x['kl_copy_max']:.5f}")
