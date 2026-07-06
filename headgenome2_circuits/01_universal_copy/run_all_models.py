import os
import subprocess
import sys

# Ensure PYTHONPATH is set correctly
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

models = ["qwen-0.5b", "qwen-1.5b", "llama-1b", "gemma-2b"]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, env=env, check=True)

for model in models:
    print(f"\n{'='*50}\nStarting pipeline for {model}\n{'='*50}")
    
    # 1. Profile Entropy
    print("\n--- 1. Profiling Entropy ---")
    runner = f"""
import sys
import importlib.util
sys.path.append('headgenome2_circuits')
spec = importlib.util.spec_from_file_location("profiler", "headgenome2_circuits/utils/head_profiler.py")
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)
profiler.profile_heads('{model}')
"""
    with open("temp_run.py", "w") as f: f.write(runner)
    run_cmd("python temp_run.py")
    
    # 2. Probe Copy Heads
    print("\n--- 2. Probing Copy Heads ---")
    runner = f"""
import sys
import importlib.util
sys.path.append('headgenome2_circuits')
spec = importlib.util.spec_from_file_location("probe", "headgenome2_circuits/01_universal_copy/probe_copy_heads.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.run_probe('{model}')
"""
    with open("temp_run.py", "w") as f: f.write(runner)
    run_cmd("python temp_run.py")
    
    # 3. Run Strict Matched Ablation
    print("\n--- 3. Running Matched Ablation ---")
    runner = f"""
import sys
import importlib.util
sys.path.append('headgenome2_circuits')
spec = importlib.util.spec_from_file_location("ablate", "headgenome2_circuits/01_universal_copy/ablate_copy_circuit.py")
ablate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablate)
ablate.run_ablation('{model}', k_heads=4)
"""
    with open("temp_run.py", "w") as f: f.write(runner)
    run_cmd("python temp_run.py")

if os.path.exists("temp_run.py"):
    os.remove("temp_run.py")

print("\n" + "="*50)
print("ALL MODELS COMPLETED.")
print("PRE-REGISTERED CRITERIA: If > 3/4 models show (Baseline == Copy Ablation), the structural illusion is universal.")
print("Check outputs/phase2_circuits/copy_ablation_*.json for final statistics.")
print("="*50 + "\n")
