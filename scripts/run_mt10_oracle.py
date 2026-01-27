import os
import subprocess
import sys

def run_oracle_mt10():
    # Oracle trains on a single task (e.g., Task 0: reach-v3)
    cmd = [
        sys.executable, "scripts/train_baseline_ppo.py",
        "--algo", "oracle",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--task-id", "0",
        "--total-timesteps", "500000",
        "--exp-name", "mt10_oracle_task0",
        "--num-envs", "4",
        "--seed", "1"
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_oracle_mt10()
