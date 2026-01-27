import os
import subprocess
import sys

def run_varshare_mt10():
    cmd = [
        sys.executable, "scripts/train_varshare_ppo.py",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "500000",
        "--exp-name", "mt10_varshare",
        "--num-envs", "4",
        "--seed", "1"
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_varshare_mt10()
