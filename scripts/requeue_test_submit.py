"""
SLURM Requeue Test Script (v3)
==============================
Purpose: Verify that the SIGUSR1 -> checkpoint -> requeue -> resume pipeline
works correctly for EVERY training script type.

IMPORTANT: MetaWorld environment initialization takes 3-5 minutes on CPU.
The time limit must be long enough for init to complete AND some training
to happen before SIGUSR1 fires.

How it works:
  1. Each job runs with a 10-minute time limit.
  2. SLURM sends SIGUSR1 at 8 minutes (--signal=B:USR1@120).
  3. By then, MetaWorld init is done and training is underway.
  4. The Python script catches the signal, saves checkpoint.pt, exits with code 99.
  5. The bash wrapper detects exit code 99 and requeues the job.
  6. On restart, the script finds checkpoint.pt, loads it, and resumes training.

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
import shutil

# We test one representative method per training script:
METHODS = {
    "det_base":    ("train_routing.py",   ["--variant", "base"]),
    "det_routing": ("train_routing.py",   ["--variant", "routing"]),
    "shared":      ("train_baselines.py", ["--algo", "shared"]),
    "pcgrad":      ("train_baselines.py", ["--algo", "pcgrad"]),
    "soft_mod":    ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    "paco":        ("train_baselines.py", ["--algo", "paco"]),
}

# 1M steps: enough that training can't finish in 5 minutes of actual compute.
# With n_steps=512, num_envs=10, that's ~195 PPO updates.
TOTAL_STEPS = 1000000
NUM_ENVS = 10
N_STEPS = 512
BATCH_SIZE = 512
EVAL_FREQ = 500000  # Avoid eval overhead during the test
HIDDEN_DIM = 256
WANDB_PROJECT = "varshare-requeue-test"

# SLURM timing:
#   MetaWorld init takes ~3-5 min on CPU.
#   We need at least 5 min for init + some training time before the signal.
#   --time=00:10:00  -> 10 minute hard limit
#   --signal=B:USR1@120 -> SIGUSR1 at the 8-minute mark (2 min before end)
#   This gives ~3-5 min for init + 3-5 min training before signal.
SLURM_TIME = "00:10:00"
SIGNAL_SPEC = "B:USR1@120"  # 120 seconds before end = signal at 8 min


def submit_all():
    user = os.environ.get("USER", "nertinger")
    base_analysis = f"/netscratch/{user}/varshare/analysis/requeue_test"
    base_logs = f"/netscratch/{user}/varshare/logs/requeue_test"
    
    # Clean up old checkpoints and logs from previous test attempts
    print("Cleaning up old requeue test data...")
    for name in METHODS:
        # Clean old analysis (checkpoint files)
        old_analysis = os.path.join(base_analysis, name)
        if os.path.exists(old_analysis):
            shutil.rmtree(old_analysis)
            print(f"  Removed: {old_analysis}")
        # Clean old logs
        old_logs = os.path.join(base_logs, name)
        if os.path.exists(old_logs):
            shutil.rmtree(old_logs)
            print(f"  Removed: {old_logs}")
    
    print()
    print("=" * 60)
    print("  SLURM REQUEUE TEST SUBMISSION (v3)")
    print(f"  Time limit: {SLURM_TIME} | Signal: USR1 at 8 min")
    print("  CPU-only (batch partition) for fast scheduling")
    print(f"  Total steps: {TOTAL_STEPS:,}")
    print("=" * 60)
    
    submitted = 0
    
    for name, (script, base_args) in METHODS.items():
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
#SBATCH --output={log_dir}/%A.out
#SBATCH --error={log_dir}/%A.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time={SLURM_TIME}
#SBATCH --signal={SIGNAL_SPEC}
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
    print(f"\nExpected timeline:")
    print(f"  0:00 - 4:00  : MetaWorld environment initialization (CPU is slow)")
    print(f"  4:00 - 8:00  : Training begins, progress lines appear in .out")
    print(f"  8:00         : SLURM sends SIGUSR1 signal")
    print(f"  8:00 - 8:05  : Python catches signal, finishes update, saves checkpoint")
    print(f"  8:05         : Python exits with code 99, bash requeues the job")
    print(f"  8:10 - 12:00 : Job restarts, loads checkpoint, resumes W&B run")
    print(f"  (repeats until all 1M steps are done)")
    print(f"\nVerification (after ~10 minutes):")
    print(f"  cat {base_logs}/det_base/*.out")
    print(f"  Look for the full sequence:")
    print(f"    [WARNING] Caught SIGUSR1 -> [CHECKPOINT] -> [REQUEUE] -> [RESUME]")

if __name__ == "__main__":
    submit_all()
