import json

with open('outputs/phase7/head_tiers_gmm.json') as f:
    tiers = json.load(f)
    
with open('outputs/phase7/head_audit.json') as f:
    audit_data = json.load(f)

# The audit json has entries for both sink and local. 
# We need to map (layer, head, type) -> error
audit = { (x["layer"], x["head"], x["type"]): x for x in audit_data }

t1 = [x for x in tiers if x["tier"] == 1]
print("Tier 1 heads errors:")
for x in t1[:10]:
    err = audit[(x["layer"], x["head"], x["type"])]["out_l_inf_natural_max"]
    print(f"L{x['layer']}H{x['head']} {x['type']}: err = {err:.4f}")
