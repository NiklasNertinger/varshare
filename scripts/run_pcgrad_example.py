import subprocess
import sys

def run_pcgrad_example():
    cmd = [
        sys.executable, "scripts/train_baseline_ppo.py",
        "--algo", "pcgrad",
        "--total-timesteps", "200000",
        "--exp-name", "pcgrad_mtl_baseline",
        "--num-envs", "8",
        "--eval-freq", "5000",
        "--eval-episodes", "5"
    ]
    
    print(f"Running PCGrad-MTL baseline (with task embeddings and gradient projection)...")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_pcgrad_example()
