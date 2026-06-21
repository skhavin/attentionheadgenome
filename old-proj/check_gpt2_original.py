
import json
import os

phase7_dir = os.path.join(os.path.dirname(__file__), "outputs", "phase7")
gpt2_audit_path = os.path.join(phase7_dir, "head_audit.json")

with open(gpt2_audit_path, "r") as f:
    gpt2_audit = json.load(f)

# Let's count how many sink/local heads had attn_l_inf_natural_mean < 0.10 (the original threshold)
print("=== GPT-2 Original Threshold Check (attn_l_inf_natural_mean < 0.10) ===")
sink_low_error = 0
local_low_error = 0
total_sink = 0
total_local = 0

for entry in gpt2_audit:
    if entry["type"] == "sink":
        total_sink +=1
        if entry["attn_l_inf_natural_mean"] < 0.10:
            sink_low_error +=1
    elif entry["type"] == "local":
        total_local +=1
        if entry["attn_l_inf_natural_mean"] < 0.10:
            local_low_error +=1

print(f"Sink heads with mean <0.10: {sink_low_error}/{total_sink}")
print(f"Local heads with mean <0.10: {local_low_error}/{total_local}")

# Let's check the first few entries of the original GPT-2 audit
print("\n=== First 20 GPT-2 Audit Entries ===")
for e in gpt2_audit[:20]:
    print(e)
