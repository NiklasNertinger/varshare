"""
SLURM Requeue Test Script
=========================
Purpose: Verify that the SIGUSR1 -> checkpoint -> requeue -> resume pipeline
works correctly for EVERY training script type.

How it works:
  1. Each job runs with a 5-minute time limit.
  2. SLURM sends SIGUSR1 at 3 minutes (--signal=B:USR1@120).
  3. The Python script catches the signal, saves checkpoint.pt, exits with code 99.
  4. The bash wrapper detects exit code 99 and requeues the job.
  5. On restart, the script finds checkpoint.pt, loads it, and resumes training.
  6. The second run completes naturally (remaining steps finish within 5 minutes).

Verification checklist (check the .out log files):
  [1] First run prints: "[WARNING] Caught SIGUSR1 from SLURM!"
  [2] First run prints: "[CHECKPOINT] Safe shutdown complete. Exiting with code 99."
  [3] Second run prints: "[RESUME] Found checkpoint at step XXX. W&B ID: ..."
  [4] Second run prints: "Loading checkpoint from ..."
  [5] Second run's global_step starts from where first run left off (not 0).
  [6] On W&B, there is exactly ONE run per method (not two fragmented runs).
"""

import os
import subprocess

# We test one representative method per training script:
#   - det_base    -> train_routing.py (simple, no gradient tricks)
#   - det_routing -> train_routing.py (GR-Share with PCGrad routing)
#   - shared      -> train_baselines.py (simplest baseline)
#   - pcgrad      -> train_baselines.py (PCGrad baseline)
#   - soft_mod    -> train_baselines.py (Soft Modularization)
#   - paco        -> train_baselines.py (PaCo)
METHODS = {
    "det_base":    ("train_routing.py",   ["--variant", "base"]),
    "det_routing": ("train_routing.py",   ["--variant", "routing"]),
    "shared":      ("train_baselines.py", ["--algo", "shared"]),
    "pcgrad":      ("train_baselines.py", ["--algo", "pcgrad"]),
    "soft_mod":    ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    "paco":        ("train_baselines.py", ["--algo", "paco"]),
}

# Use a small step count so the SECOND run finishes naturally within 5 minutes.
# 100k steps is enough to run for ~3 min on CPU, get interrupted, then finish the rest.
TOTAL_STEPS = 100000
NUM_ENVS = 10
N_STEPS = 512
BATCH_SIZE = 512
EVAL_FREQ = 50000
HIDDEN_DIM = 256
WANDB_PROJECT = "varshare-requeue-test"

def submit_all():
    user = os.environ.get("USER", "nertinger")
    
    print("=" * 60)
    print("  SLURM REQUEUE TEST SUBMISSION")
    print("  Time limit: 5 min | Signal: USR1 at 3 min")
    print("  CPU-only (batch partition) for fast scheduling")
    print("=" * 60)
    
    submitted = 0
    
    for name, (script, base_args) in METHODS.items():
        # Build the analysis dir so checkpoints go to a unique, clean location
        analysis_dir = f"/netscratch/{user}/varshare/analysis/requeue_test/{name}"
        log_dir = f"/netscratch/{user}/varshare/logs/requeue_test/{name}"
        os.makedirs(log_dir, exist_ok=True)
        
        config_args = [
            "--exp-name", f"requeue_test_{name}",
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
#SBATCH --job-name=rq_{name}
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --signal=B:USR1@120

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

echo "========================================"
echo "  REQUEUE TEST: {name}"
echo "  Started at: $(date)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Restart count: $SLURM_RESTART_COUNT"
echo "========================================"

{full_cmd}
EXIT_CODE=$?

echo "Script exited with code: $EXIT_CODE"

if [ $EXIT_CODE -eq 99 ]; then
    echo "[REQUEUE] Caught exit code 99. Requeueing job $SLURM_JOB_ID..."
    scontrol requeue $SLURM_JOB_ID
else
    echo "[DONE] Job completed normally with exit code $EXIT_CODE."
fi
"""
        
        script_path = f"logs/temp_requeue_test_{name}.sh"
        os.makedirs("logs", exist_ok=True)
        with open(script_path, "w") as f:
            f.write(sbatch_script)
        
        print(f"\nSubmitting requeue test: {name}")
        print(f"  Script: {script}")
        print(f"  Args: {' '.join(base_args)}")
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
        submitted += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Submitted {submitted} requeue test jobs.")
    print(f"  W&B Project: {WANDB_PROJECT}")
    print(f"  Monitor logs: /netscratch/{user}/varshare/logs/requeue_test/")
    print(f"{'=' * 60}")
    print(f"\nVerification steps:")
    print(f"  1. Wait ~3 min for first SIGUSR1 trigger")
    print(f"  2. Check logs for '[CHECKPOINT] Safe shutdown complete'")
    print(f"  3. Check SLURM queue for requeued jobs: squeue -u $USER")
    print(f"  4. After requeue, check logs for '[RESUME] Found checkpoint'")
    print(f"  5. Check W&B project '{WANDB_PROJECT}' for single continuous runs")

if __name__ == "__main__":
    submit_all()
