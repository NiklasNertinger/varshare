import subprocess
import sys
import os

def run_soft_mod():
    # Parameters for Soft Modularization baseline
    args = [
        sys.executable,
        "scripts/train_baseline_ppo.py",
        "--algo", "soft_mod",
        "--num-modules", "4",
        "--total-timesteps", "200000",
        "--exp-name", "soft_mod_cartpole_5task",
        "--num-envs", "4",
        "--lr-actor", "0.001",
        "--lr-critic", "0.0001",
        "--ent-coef", "0.01"
    ]
    
    print(f"Running Soft Modularization Baseline: {' '.join(args)}")
    subprocess.run(args)

if __name__ == "__main__":
    run_soft_mod()
