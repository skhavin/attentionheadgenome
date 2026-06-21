import json
import numpy as np

with open("outputs/phase7/head_audit.json") as f:
    data = json.load(f)

print("Total entries:", len(data))

# Since the previous run stored out_l_inf_rel (which was relative_linf), we can't easily reconstruct the absolute L-infinity.
# Wait, let's look at the fields in head_audit.json:
# 'attn_l_inf_natural_max', 'out_l_inf_rel_natural_max', 'kl_natural_max', etc.
# Wait, we need to run a small check. Let's see what the range of out_l_inf_rel is.

for output_linf_thresh in [0.015, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]:
    for attn_linf_thresh in [0.001, 0.01, 0.05, 0.10, 0.50, 1.0]:
        for check_attn_on_local in [True, False]:
            t1, t2, t3 = 0, 0, 0
            for x in data:
                kl_nat = x["kl_natural_max"]
                kl_copy = x["kl_copy_max"]
                out_nat = x["out_l_inf_rel_natural_max"]
                out_copy = x["out_l_inf_rel_copy_max"]
                attn_nat = x["attn_l_inf_natural_max"]
                attn_copy = x["attn_l_inf_copy_max"]
                
                # Check if we should ignore attn_linf on local
                is_local = (x["type"] == "local")
                if is_local and not check_attn_on_local:
                    attn_ok_nat = True
                    attn_ok_copy = True
                else:
                    attn_ok_nat = (attn_nat < attn_linf_thresh)
                    attn_ok_copy = (attn_copy < attn_linf_thresh)
                
                nat_safe = (attn_ok_nat and out_nat < output_linf_thresh and kl_nat < 0.010)
                copy_safe = (attn_ok_copy and out_copy < output_linf_thresh and kl_copy < 0.010)
                
                if nat_safe and copy_safe:
                    t1 += 1
                elif nat_safe and not copy_safe:
                    t2 += 1
                else:
                    t3 += 1
            
            # Print if we get any Tier 1
            if t1 > 0:
                print(f"out_thresh={output_linf_thresh:.3f}, attn_thresh={attn_linf_thresh:.3f}, check_attn_local={check_attn_on_local} -> Tier 1: {t1} ({t1/len(data)*100:.1f}%), Tier 2: {t2} ({t2/len(data)*100:.1f}%), Tier 3: {t3}")
