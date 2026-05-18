"""
CPU vs GPU Speed Benchmark
==========================
Runs all core methods for 100k steps on both CPU and GPU partitions.
All runs log to the same W&B project for easy comparison.

Each run is tagged with 'cpu' or 'gpu' so you can filter/group in W&B.
"""

import os
import subprocess

METHODS = {
    "det_base":    ("train_routing.py",   ["--variant", "base"]),
    "det_routing": ("train_routing.py",   ["--variant", "routing"]),
    "shared":      ("train_baselines.py", ["--algo", "shared"]),
    "pcgrad":      ("train_baselines.py", ["--algo", "pcgrad"]),
    "soft_mod":    ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    "paco":        ("train_baselines.py", ["--algo", "paco"]),
}

TOTAL_STEPS = 100000
NUM_ENVS = 10
N_STEPS = 512
BATCH_SIZE = 512
EVAL_FREQ = 50000
HIDDEN_DIM = 256
WANDB_PROJECT = "varshare-cpu-vs-gpu"

CONFIGS = {
    "cpu": {
        "partition": "batch",
        "gres": "",
        "time": "01:00:00",
        "extra_sbatch": "",
    },
    "gpu": {
        "partition": "RTX3090",
        "gres": "#SBATCH --gres=gpu:1",
        "time": "00:30:00",
        "extra_sbatch": "",
    },
}


def submit_all():
    user = os.environ.get("USER", "nertinger")

    print("=" * 60)
    print("  CPU vs GPU SPEED BENCHMARK")
    print(f"  Steps: {TOTAL_STEPS:,} | W&B: {WANDB_PROJECT}")
    print("=" * 60)

    submitted = 0

    for device_tag, cfg in CONFIGS.items():
        for name, (script, base_args) in METHODS.items():
            run_label = f"{name}_{device_tag}"
            analysis_dir = f"/netscratch/{user}/varshare/analysis/speed_bench/{run_label}"
            log_dir = f"/netscratch/{user}/varshare/logs/speed_bench/{run_label}"
            os.makedirs(log_dir, exist_ok=True)

            config_args = [
                "--exp-name", f"speed_{run_label}",
                "--total-timesteps", str(TOTAL_STEPS),
                "--n-steps", str(N_STEPS),
                "--batch-size", str(BATCH_SIZE),
                "--num-envs", str(NUM_ENVS),
                "--eval-mode", "True",
                "--eval-freq", str(EVAL_FREQ),
                "--env-type", "metaworld",
                "--mt-setting", "MT10",
                "--hidden-dim", str(HIDDEN_DIM),
                "--wandb-project", WANDB_PROJECT,
                "--analysis-dir", analysis_dir,
            ]

            all_args = base_args + config_args
            full_cmd = f"python {script} " + " ".join(all_args) + " --seed 1"

            sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=spd_{run_label[:12]}
#SBATCH --output={log_dir}/%A.out
#SBATCH --error={log_dir}/%A.err
#SBATCH --partition={cfg['partition']}
{cfg['gres']}
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time={cfg['time']}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
export PYTHONUNBUFFERED=1

echo "========================================"
echo "  SPEED BENCHMARK: {name} ({device_tag.upper()})"
echo "  Started at: $(date)"
echo "========================================"

{full_cmd}
"""

            script_path = f"logs/temp_speed_{run_label}.sh"
            os.makedirs("logs", exist_ok=True)
            with open(script_path, "w") as f:
                f.write(sbatch_script)

            print(f"  [{device_tag.upper():>3}] {name:<15} ", end="")
            result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
            print(result.stdout.strip())
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr.strip()}")
            submitted += 1

    print(f"\n{'=' * 60}")
    print(f"  Submitted {submitted} jobs ({len(METHODS)} methods x 2 devices)")
    print(f"  W&B Project: {WANDB_PROJECT}")
    print(f"  Group by exp_name or filter by 'cpu'/'gpu' in the run name")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    submit_all()
