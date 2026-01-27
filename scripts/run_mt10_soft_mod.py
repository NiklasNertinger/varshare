import os
import subprocess
import sys

def run_soft_mod_mt10():
    cmd = [
        sys.executable, "scripts/train_baseline_ppo.py",
        "--algo", "soft_mod",
        "--num-modules", "4",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "500000",
        "--exp-name", "mt10_soft_mod",
        "--num-envs", "4",
        "--seed", "1"
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_soft_mod_mt10()
