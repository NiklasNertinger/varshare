import os
import pathlib
import pandas as pd
from datetime import datetime

def monitor_runs():
    base_dir = pathlib.Path("/netscratch") / os.environ.get("USER", "default_user") / "varshare" / "analysis" / "final_eval"
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist yet. No runs started?")
        return
        
    runs = []
    
    # Target Steps for completion percentage
    t_steps = {
        "CP": 2000000,
        "LL": 10000000,
        "MT4": 25000000,
        "MT10": 50000000
    }
    
    for csv_file in base_dir.rglob("heartbeat.csv"):
        try:
            # We only read the last line to save memory/speed using bash tail
            import subprocess
            res = subprocess.run(["tail", "-n", "1", str(csv_file)], capture_output=True, text=True)
            if not res.stdout.strip(): continue
            
            # Read header to get column logic
            with open(csv_file, 'r') as f:
                header = f.readline().strip().split(',')
                
            last_line = res.stdout.strip().split(',')
            if len(header) != len(last_line):
                continue
                
            data = dict(zip(header, last_line))
            
            # Extract path details
            # path: .../phaseX/ENV/ALGO/seed_Y/heartbeat.csv
            parts = csv_file.parts
            seed = parts[-2]
            algo = parts[-3]
            env = parts[-4]
            phase = parts[-5]
            
            # Math
            step = float(data.get("TOTAL_ENV_STEPS", 0))
            target = float(t_steps.get(env, 1)) # Defaults to 1 if env not found to avoid ZeroDivisionError
            pct = (step / target) * 100
            
            reward = float(data.get("eval/mean_reward", 0.0))
            sps = float(data.get("SPS", 0.0))
            
            # Hours remaining approx
            rem_steps = target - step
            rem_hours = (rem_steps / sps) / 3600 if sps > 0 else 0
            
            # Formatting
            if rem_hours < 0:
                rem_hours = 0.0
                
            runs.append({
                "Env": env,
                "Alg": algo,
                "Seed": seed,
                "Pct": pct,
                "Step": f"{step/1000000:.2f}M",
                "Rew": reward,
                "SPS": int(sps),
                "ETA": f"{rem_hours:.1f}h"
            })
            
        except Exception as e:
            pass

    if not runs:
        print("No heartbeat data found in final_eval directory.")
        return

    # Sort
    runs = sorted(runs, key=lambda x: (x["Env"], x["Alg"], x["Seed"]))
    
    # Print formatted table
    print("=" * 95)
    print(f"{'ENVIRONMENT':<12} | {'ALGORITHM':<30} | {'SEED':<6} | {'%':<6} | {'STEPS':<8} | {'REWARD':<8} | {'SPS':<6} | {'ETA':<8}")
    print("-" * 95)
    
    last_env = ""
    for r in runs:
        if r["Env"] != last_env and last_env != "":
            print("-" * 105)
        last_env = r["Env"]
        
        # Format Algorithm name to fit the 30-char column cleanly without breaking alignment
        disp_alg = r['Alg']
        if len(disp_alg) > 30:
            disp_alg = disp_alg[:27] + "..."
            
        print(f"{r['Env']:<12} | {disp_alg:<30} | {r['Seed']:<6} | {r['Pct']:>6.2f}% | {r['Step']:>8} | {r['Rew']:>8.1f} | {r['SPS']:>6} | {r['ETA']:>8}")

    print("=" * 105)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_runs()
