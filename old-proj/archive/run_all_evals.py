import subprocess
import sys

ratios = ["0.75", "0.50", "0.25", "0.10"]
python_path = "/home/zeus/miniconda3/envs/cloudspace/bin/python"
wrapper_path = "/teamspace/studios/this_studio/evaluation.py"

for ratio in ratios:
    cmd = [
        python_path,
        wrapper_path,
        "--method", "proactive_cache",
        "--compression_ratio", ratio,
        "--dataset", "ruler",
        "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ]
    print(f"Starting run for ratio {ratio}...")
    print(f"Command: {' '.join(cmd)}")
    sys.stdout.flush()
    
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Error: Run failed for ratio {ratio} with exit code {ret.returncode}")
        sys.exit(ret.returncode)
    print(f"Successfully completed run for ratio {ratio}.\n")
    sys.stdout.flush()

print("All four ratios evaluated successfully!")
