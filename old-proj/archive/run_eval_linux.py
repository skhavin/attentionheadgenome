import os
import sys
import subprocess

# Add kvpress to path so it can be imported
sys.path.insert(0, "/teamspace/studios/this_studio/kvpress")

# Import shim
import shlex
sys.modules['pipes'] = shlex

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/teamspace/studios/this_studio/.cache/huggingface"

# Change directory to the evaluation directory so evaluate.py runs in context
os.chdir("/teamspace/studios/this_studio/kvpress/evaluation")

# Set PYTHONPATH to prioritize our local kvpress directory
os.environ["PYTHONPATH"] = "/teamspace/studios/this_studio/kvpress:" + os.environ.get("PYTHONPATH", "")

# Execute evaluate.py
cmd = [
    "/home/zeus/miniconda3/envs/cloudspace/bin/python",
    "-c",
    "import sys, shlex; sys.modules['pipes'] = shlex; import runpy; runpy.run_path('evaluate.py', run_name='__main__')"
] + sys.argv[1:]

print(f"Running command: {' '.join(cmd)}")
sys.exit(subprocess.run(cmd).returncode)
