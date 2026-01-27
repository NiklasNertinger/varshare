import subprocess
import sys
import os

def run_paco():
    # Parameters for PaCo baseline
    args = [
        sys.executable,
        "scripts/train_baseline_ppo.py",
        "--algo", "paco",
        "--num-experts", "4",
        "--total-timesteps", "200000",
        "--exp-name", "paco_cartpole_5task",
        "--num-envs", "4",
        "--lr-actor", "0.001",
        "--lr-critic", "0.0001",
        "--ent-coef", "0.01"
    ]
    
    print(f"Running PaCo Baseline: {' '.join(args)}")
    subprocess.run(args)

if __name__ == "__main__":
    run_paco()
