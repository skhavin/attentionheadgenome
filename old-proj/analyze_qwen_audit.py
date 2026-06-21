
import json
import os
import numpy as np
from collections import defaultdict

# Load the Qwen audit data
phase7_dir = os.path.join(os.path.dirname(__file__), "outputs", "phase7")
audit_path = os.path.join(phase7_dir, "qwen_head_audit.json")

with open(audit_path, "r") as f:
    audit_data = json.load(f)

# Organize by head type
sink_data = defaultdict(list)
local_data = defaultdict(list)

for entry in audit_data:
    if entry["type"] == "sink":
        sink_data["attn_linf_nat"].append(entry["attn_l_inf_natural_max"])
        sink_data["kl_nat"].append(entry["kl_natural_max"])
        sink_data["kl_copy"].append(entry["kl_copy_max"])
    elif entry["type"] == "local":
        local_data["out_linf_nat"].append(entry["out_l_inf_natural_max"])
        local_data["kl_nat"].append(entry["kl_natural_max"])
        local_data["kl_copy"].append(entry["kl_copy_max"])

print("=== Sink Head Errors ===")
for key, vals in sink_data.items():
    print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}, 50th={np.percentile(vals, 50):.4f}, 90th={np.percentile(vals, 90):.4f}")

print("\n=== Local Head Errors ===")
for key, vals in local_data.items():
    print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}, 50th={np.percentile(vals, 50):.4f}, 90th={np.percentile(vals, 90):.4f}")

# Let's also look at the original GPT-2 audit to compare!
gpt2_audit_path = os.path.join(phase7_dir, "head_audit.json")
if os.path.exists(gpt2_audit_path):
    with open(gpt2_audit_path, "r") as f:
        gpt2_audit = json.load(f)
        
    print("\n\n=== GPT-2 Sink Head Errors ===")
    gpt2_sink = [e for e in gpt2_audit if e["type"] == "sink"]
    for key in ["attn_l_inf_natural_max", "kl_natural_max", "kl_copy_max"]:
        vals = [e[key] for e in gpt2_sink if e[key] is not None]
        print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}, 50th={np.percentile(vals, 50):.4f}, 90th={np.percentile(vals, 90):.4f}")
        
    print("\n=== GPT-2 Local Head Errors ===")
    gpt2_local = [e for e in gpt2_audit if e["type"] == "local"]
    for key in ["out_l_inf_natural_max", "kl_natural_max", "kl_copy_max"]:
        vals = [e[key] for e in gpt2_local if e[key] is not None]
        print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}, 50th={np.percentile(vals, 50):.4f}, 90th={np.percentile(vals, 90):.4f}")
