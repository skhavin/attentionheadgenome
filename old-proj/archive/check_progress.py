# Script to monitor the active PG-19 benchmark progress and report results when done.
import os
import subprocess
import pickle
import time
import sys

def get_process_info():
    """Find the running benchmark_pg19.py process and return its details."""
    try:
        # Run wmic to find python processes with benchmark_pg19 in command line
        cmd = 'wmic process where "CommandLine like \'%benchmark_pg19%\'" get ProcessId,CommandLine,CreationDate /format:csv'
        out = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        lines = [line.strip().split(',') for line in out.strip().split('\n') if line.strip()]
        
        # Parse output
        valid_processes = []
        for cols in lines:
            if len(cols) >= 4 and "wmic" not in cols[1] and "check_progress" not in cols[1]:
                # Format: Node, CommandLine, CreationDate, ProcessId
                pid = cols[3].strip()
                if pid.isdigit():
                    valid_processes.append({
                        "pid": int(pid),
                        "cmd": cols[1],
                        "created": cols[2]
                    })
        return valid_processes
    except Exception:
        return []

def main():
    print("=" * 60)
    print("                PG-19 BENCHMARK PROGRESS MONITOR                ")
    print("=" * 60)

    # 1. Check if process is currently running
    procs = get_process_info()
    if procs:
        print(f"\n[ACTIVE] STATUS: Active & Running!")
        for p in procs:
            print(f"  * Process ID (PID): {p['pid']}")
            print(f"  * Command Line:     {p['cmd']}")
            
            # Get resource stats via PowerShell
            try:
                ps_cmd = f'powershell -Command "Get-Process -Id {p["pid"]} | Select-Object CPU, @{{Name=\'MemMB\'; Expression={{[math]::Round($_.PM/1MB,1)}}}} | ConvertTo-Json"'
                stats_out = subprocess.check_output(ps_cmd, shell=True).decode('utf-8')
                import json
                stats = json.loads(stats_out)
                print(f"  * CPU Time Used:    {stats.get('CPU', 'N/A')} seconds")
                print(f"  * System RAM Used:   {stats.get('MemMB', 'N/A')} MB")
            except Exception:
                pass
                
            try:
                # Check VRAM allocation
                nv_cmd = 'nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits'
                nv_out = subprocess.check_output(nv_cmd, shell=True).decode('utf-8').strip().split(',')
                if len(nv_out) >= 3:
                    print(f"  * GPU VRAM Usage:   {nv_out[2].strip()} MB / {nv_out[0].strip()} MB")
            except Exception:
                pass
        
        print("\n[TIP] Tip: The benchmark evaluates methods locally on the GPU. It saves results to output file when finished.")
        print("    Run this script again to monitor progress.")
    else:
        print("\n[STOPPED] STATUS: Not currently running (completed or stopped).")
        
        # 2. Check if output pickle exists
        pkl_path = "outputs/phase5/pg19_benchmark.pkl"
        if os.path.exists(pkl_path):
            print(f"\n[SUCCESS] SUCCESS: Final benchmark results found at '{pkl_path}'!")
            try:
                with open(pkl_path, "rb") as f:
                    results = pickle.load(f)
                
                print("\n=== PG-19 Benchmark Results Table ===\n")
                print(f"{'Method':<20} {'Budget':>6} {'PPL':>8} {'VRAM(MB)':>9} {'Time(s)':>8}")
                print("-" * 56)
                if "full" in results:
                    r = results["full"]
                    print(f"{'Full Attention':<20} {'all':>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")
                
                budgets = [128, 256, 512, 1024]
                for budget in budgets:
                    print("-" * 56)
                    for method, label in [("streamingllm", "StreamingLLM"), ("h2o", "H2O"), ("proactive", "Proactive (ours)")]:
                        key = f"{method}_{budget}"
                        if key in results:
                            r = results[key]
                            print(f"{label:<20} {budget:>6} {r['ppl']:>8.2f} {r['vram_mb']:>9.0f} {r['time_s']:>8.1f}")
                print("-" * 56)
            except Exception as e:
                print(f"Error reading results: {e}")
        else:
            print(f"\n[PENDING] No results pickle file found at '{pkl_path}' yet.")
            print("   If you just started it, please allow a few minutes for the first runs to compile.")

if __name__ == "__main__":
    main()
