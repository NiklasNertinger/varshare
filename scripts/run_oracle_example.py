import subprocess
import sys

def run_oracle_example():
    cmd = [
        sys.executable, "scripts/train_baseline_ppo.py",
        "--algo", "oracle",
        "--task-id", "0",
        "--total-timesteps", "200000",
        "--exp-name", "oracle_baseline_task0",
        "--num-envs", "8",
        "--eval-freq", "5000",
        "--eval-episodes", "5"
    ]
    
    print(f"Running Oracle baseline on Task 0...")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_oracle_example()
