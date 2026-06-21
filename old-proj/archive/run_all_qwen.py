import subprocess
import sys
import os

model = "Qwen/Qwen2.5-0.5B"

phases = [
    # Phase 1 is already running separately, so we comment it out, but keep it for reference
    # ["python", "phase7/audit_heads.py", "--model", model],
    ["python", "phase7/regime_detector.py", "--model", model],
    ["python", "phase7/eval_ppl.py", "--model", model, "--dataset", "wikitext"],
    ["python", "phase7/eval_ppl.py", "--model", model, "--dataset", "pg19"],
    ["python", "phase7/eval_ppl.py", "--model", model, "--dataset", "induction"],
    ["python", "phase7/eval_ruler.py", "--model", model],
    ["python", "phase7/eval_downstream.py", "--model", model]
]

for cmd in phases:
    print(f"\n{'='*80}\nRunning: {' '.join(cmd)}\n{'='*80}")
    sys.stdout.flush()
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"\nError: Command failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

print("\nAll phases completed successfully!")
