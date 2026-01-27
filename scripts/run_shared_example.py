import subprocess
import sys

def run_shared_example():
    cmd = [
        sys.executable, "scripts/train_baseline_ppo.py",
        "--algo", "shared",
        "--total-timesteps", "200000",
        "--exp-name", "shared_mtl_baseline",
        "--num-envs", "8",
        "--eval-freq", "5000",
        "--eval-episodes", "5"
    ]
    
    print(f"Running Shared-MTL baseline (with task embeddings)...")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_shared_example()
