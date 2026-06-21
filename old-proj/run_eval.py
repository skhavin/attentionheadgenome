"""
Launcher: runs kvpress evaluation.py from its correct working directory.
Sets the HuggingFace cache directory to D: drive and shims pipes module for Python 3.13+.
Usage: python run_eval.py --press_name proactive_cache_75 --dataset ruler
"""
import os, sys, subprocess, json

# 1. Setup paths
evaluation_dir = os.path.join(os.path.dirname(__file__), "..", "kvpress", "evaluation")
evaluation_dir = os.path.abspath(evaluation_dir)

args = sys.argv[1:]  # pass through all CLI args

# If --model_kwargs is not specified, add it with {"device_map": {"": "cuda"}} to prevent CPU offloading
has_model_kwargs = any(a.startswith("--model_kwargs") for a in args)
if not has_model_kwargs:
    # We pass it as a JSON string for Fire to parse
    args += ["--model_kwargs", '{"device_map": {"": "cuda"}}']

# 2. Run evaluate.py as a script
cmd = [
    sys.executable,
    "-c",
    "import sys, shlex; sys.modules['pipes'] = shlex; "
    "import runpy; runpy.run_path('evaluate.py', run_name='__main__')"
] + args

print(f"Running from: {evaluation_dir}")
print(f"Command: {' '.join(cmd)}\n")

# 3. Setup environment variables
env = os.environ.copy()
env["PYTHONPATH"] = evaluation_dir + os.pathsep + env.get("PYTHONPATH", "")
env["HF_HOME"] = r"d:\.cache\huggingface"
env["HF_HUB_DISABLE_DISK_SPACE_WARNING"] = "1"
env["SAFETENSORS_FAST_GPU"] = "1"

result = subprocess.run(cmd, cwd=evaluation_dir, env=env)
sys.exit(result.returncode)
