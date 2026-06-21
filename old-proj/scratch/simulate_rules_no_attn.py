import json

with open("outputs/phase7/head_audit.json") as f:
    data = json.load(f)

# Since the previous run stored out_l_inf_rel (relative error), we can't get absolute out_linf directly.
# But wait! We can estimate absolute out_linf if we run test_absolute_thresholds.py.
# Actually, let's run test_absolute_thresholds.py with a rule that ignores attn_linf and see the result!
