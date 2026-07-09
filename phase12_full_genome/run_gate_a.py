import os
import subprocess

print("=========================================")
print("STARTING FULL GENOME GATE A SWEEP")
print("=========================================")

print("\n--- Running 03: QK vs OV Ablation ---")
subprocess.run(["python", "03_qk_vs_ov_ablation.py"])

print("\n--- Running 04: Shuffle Survival ---")
subprocess.run(["python", "04_shuffle_survival.py"])

print("\n=========================================")
print("GATE A SWEEP COMPLETE!")
print("=========================================")
