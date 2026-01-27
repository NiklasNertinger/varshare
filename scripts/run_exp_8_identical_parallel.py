
import subprocess
import sys

seeds = [1, 2, 3]
for seed in seeds:
    print(f"Running Exp 8 (Identical Parallel) | Seed {seed}")
    subprocess.run([sys.executable, "scripts/train_varshare_ppo.py", 
                    "--exp-name", "exp8_identical_parallel", 
                    "--env-type", "IdenticalCartPole",
                    "--num-envs", "8",
                    "--n-steps", "256",
                    "--lr-actor", "0.005", "--lr-critic", "0.0005",
                    "--total-timesteps", "200000",
                    "--seed", str(seed)])
