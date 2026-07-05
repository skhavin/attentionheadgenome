import subprocess
import sys

MODELS = [
    "unsloth/Llama-3.2-1B"
]

SCRIPTS = [
    "phase2_atlas/step1_distance_profile.py",
    "phase2_atlas/step2_ov_output_norm.py",
    "phase2_atlas/step3_grammar_map.py",
    "phase2_atlas/step4_softmax_saturation.py",
    "phase2_atlas/step5_sink_falsification.py",
    "phase2_atlas/step6_compile_atlas.py"
]

def run_command(cmd):
    print(f"\n========================================")
    print(f"Executing: {' '.join(cmd)}")
    print(f"========================================")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)

for model in MODELS:
    print(f"\n\n{'#'*60}")
    print(f"# PROCESSING MODEL: {model}")
    print(f"{'#'*60}")
    for script in SCRIPTS:
        run_command(["python", script, model])

print("\n\nALL MODELS COMPLETED SUCCESSFULLY.")
