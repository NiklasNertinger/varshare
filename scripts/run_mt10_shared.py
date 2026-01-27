import os
import subprocess
import sys

def run_shared_mt10():
    cmd = [
        sys.executable, "scripts/train_baseline_ppo.py",
        "--algo", "shared",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "500000",
        "--exp-name", "mt10_shared",
        "--num-envs", "4",
        "--seed", "1"
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_shared_mt10()
