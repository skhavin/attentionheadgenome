import os
import subprocess
import sys

# Ensure PYTHONPATH is set correctly
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

models = ["qwen-0.5b", "llama-1b", "gemma-2b"]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, env=env, check=True)

for model in models:
    print(f"\n{'='*50}\nStarting pipeline for {model}\n{'='*50}")
    
    # 1. Profile Entropy
    print("\n--- 1. Profiling Entropy ---")
    profiler_cmd = f"python headgenome2_circuits/utils/head_profiler.py"
    # We need to modify head_profiler to take a model argument or patch it here
    # A quick inline script to call the function:
    runner = f"""
import sys
sys.path.append('headgenome2_circuits')
from utils.head_profiler import profile_heads
profile_heads('{model}')
"""
    with open("temp_run.py", "w") as f:
        f.write(runner)
    run_cmd("python temp_run.py")
    
    # 2. Probe Copy Heads
    print("\n--- 2. Probing Copy Heads ---")
    runner = f"""
import sys
sys.path.append('headgenome2_circuits')
from 01_universal_copy.probe_copy_heads import run_probe
run_probe('{model}')
"""
    with open("temp_run.py", "w") as f:
        f.write(runner)
    run_cmd("python temp_run.py")
    
    # 3. Run Strict Matched Ablation
    print("\n--- 3. Running Matched Ablation ---")
    runner = f"""
import sys
sys.path.append('headgenome2_circuits')
from 01_universal_copy.ablate_copy_circuit import run_ablation
run_ablation('{model}', k_heads=4)
"""
    with open("temp_run.py", "w") as f:
        f.write(runner)
    run_cmd("python temp_run.py")

if os.path.exists("temp_run.py"):
    os.remove("temp_run.py")
print("All models completed!")
