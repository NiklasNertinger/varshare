"""
SLURM Requeue Test Script
=========================
Purpose: Verify that the SIGUSR1 -> checkpoint -> requeue -> resume pipeline
works correctly for EVERY training script type.

How it works:
  1. Each job runs with a 2-minute time limit.
  2. SLURM sends SIGUSR1 at 1 minute (--signal=B:USR1@60).
  3. The Python script catches the signal, saves checkpoint.pt, exits with code 99.
  4. The bash wrapper detects exit code 99 and requeues the job.
  5. On restart, the script finds checkpoint.pt, loads it, and resumes training.
  6. The second run completes naturally (remaining steps finish within 2 minutes).

Verification checklist (check the .out log files):
  [1] First run prints: "[WARNING] Caught SIGUSR1 from SLURM!"
  [2] First run prints: "[CHECKPOINT] Safe shutdown complete. Exiting with code 99."
  [3] SLURM wrapper prints: "[REQUEUE] Caught exit code 99. Requeueing job..."
  [4] Second run prints: "[RESUME] Found checkpoint at step XXX. W&B ID: ..."
  [5] Second run prints: "Loading checkpoint from ..."
  [6] Second run's global_step starts from where first run left off (not 0).
  [7] On W&B, there is exactly ONE run per method (not two fragmented runs).
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

# Use enough steps that the job CAN'T finish in 1 minute on CPU.
# With n_steps=512, num_envs=10, each update = 5120 steps.
# 1M steps = ~195 updates. On CPU with MetaWorld, this takes ~10+ minutes.
TOTAL_STEPS = 1000000
NUM_ENVS = 10
N_STEPS = 512
BATCH_SIZE = 512
EVAL_FREQ = 500000  # Eval once at the end, not during the test
HIDDEN_DIM = 256
WANDB_PROJECT = "varshare-requeue-test"

def submit_all():
    user = os.environ.get("USER", "nertinger")
    base_analysis = f"/netscratch/{user}/varshare/analysis/requeue_test"
    base_logs = f"/netscratch/{user}/varshare/logs/requeue_test"
    
    # Clean up old checkpoints and logs from previous test attempts
    print("Cleaning up old requeue test data...")
    for name in METHODS:
        old_analysis = os.path.join(base_analysis, name)
        old_logs = os.path.join(base_logs, name)
        # Remove old checkpoint files so we get a fresh test
        checkpoint_glob = os.path.join(old_analysis, "requeue_test_" + name, "seed_1", "checkpoint.pt")
        if os.path.exists(checkpoint_glob):
            os.remove(checkpoint_glob)
            print(f"  Removed old checkpoint: {checkpoint_glob}")
    
    print()
    print("=" * 60)
    print("  SLURM REQUEUE TEST SUBMISSION")
    print("  Time limit: 2 min | Signal: USR1 at 1 min")
    print("  CPU-only (batch partition) for fast scheduling")
    print("  Total steps: 1,000,000 (guarantees >1 min runtime)")
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
        
        # Key SLURM settings for the test:
        #   --time=00:02:00       : 2-minute time limit
        #   --signal=B:USR1@60    : Send SIGUSR1 60 seconds before the end (= at 1 min)
        #   --open-mode=append    : Append to log files on requeue (don't overwrite!)
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=rq_{name}
#SBATCH --output={log_dir}/%A.out
#SBATCH --error={log_dir}/%A.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --signal=B:USR1@60
#SBATCH --open-mode=append

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
export PYTHONUNBUFFERED=1

echo ""
echo "========================================"
echo "  REQUEUE TEST: {name}"
echo "  Started at: $(date)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Restart count: ${{SLURM_RESTART_COUNT:-0}}"
echo "========================================"

{full_cmd}
EXIT_CODE=$?

echo ""
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
    print(f"  Monitor logs: {base_logs}/")
    print(f"{'=' * 60}")
    print(f"\nTimeline:")
    print(f"  0:00 - Job starts, initializes MetaWorld environments")
    print(f"  1:00 - SLURM sends SIGUSR1 signal")
    print(f"  1:00-1:05 - Python catches signal, finishes current update, saves checkpoint")
    print(f"  1:05 - Python exits with code 99, bash requeues the job")
    print(f"  1:10 - Job restarts, finds checkpoint, resumes W&B run")
    print(f"  3:00 - Second run hits 2-min limit, gets another SIGUSR1")
    print(f"         (this will repeat until all 1M steps are done)")
    print(f"\nVerification:")
    print(f"  cat /netscratch/{user}/varshare/logs/requeue_test/det_base/*.out")
    print(f"  Look for: [WARNING] Caught SIGUSR1 -> [CHECKPOINT] -> [REQUEUE] -> [RESUME]")

if __name__ == "__main__":
    submit_all()
