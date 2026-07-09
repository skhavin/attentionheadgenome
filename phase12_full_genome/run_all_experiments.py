import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*80}\nRunning {script_name}...\n{'='*80}")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}:\n{result.stderr}")
        return False
    print(f"Success! Output snippet:\n{result.stdout[-1000:]}")
    return True

if __name__ == "__main__":
    scripts = [
        "03_qk_vs_ov_ablation.py",
        "04_shuffle_survival.py",
        "07_nested_cv_causal.py"
    ]
    for s in scripts:
        if not run_script(s):
            break
    print("\nALL EXPERIMENTS COMPLETED.")
