"""
SLURM Requeue Test Script (v4)
==============================
Purpose: Verify that the SIGUSR1 -> checkpoint -> requeue -> resume pipeline
works correctly for EVERY training script type.

IMPORTANT: MetaWorld environment initialization takes 3-5 minutes on CPU.
The time limit must be long enough for init to complete AND some training
to happen before SIGUSR1 fires.

Signal routing:
  SLURM --signal=B:USR1@120 sends the signal to the BATCH SHELL only.
  We use a bash `trap` to forward SIGUSR1 to the Python child process.
  Python catches it via its own signal handler, checkpoints, and exits 99.

How it works:
  1. Each job runs with a 10-minute time limit.
  2. SLURM sends SIGUSR1 to bash at 8 minutes (--signal=B:USR1@120).
  3. Bash trap forwards SIGUSR1 to the Python child process.
  4. Python catches SIGUSR1, saves checkpoint.pt, exits with code 99.
  5. The bash wrapper detects exit code 99 and requeues the job.
  6. On restart, the script finds checkpoint.pt, loads it, and resumes training.

Verification:
  After the test completes, check requeue_verification.txt in the analysis dir.
  It contains a [SAVE] block and a [RESUME] block with matching param checksums
  and normalization stats, proving exact state continuity.
"""

import os
import subprocess
import shutil

METHODS = {
    "det_base":    ("train_routing.py",   ["--variant", "base"]),
    "det_routing": ("train_routing.py",   ["--variant", "routing"]),
    "shared":      ("train_baselines.py", ["--algo", "shared"]),
    "pcgrad":      ("train_baselines.py", ["--algo", "pcgrad"]),
    "soft_mod":    ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    "paco":        ("train_baselines.py", ["--algo", "paco"]),
}

TOTAL_STEPS = 1000000
NUM_ENVS = 10
N_STEPS = 512
BATCH_SIZE = 512
EVAL_FREQ = 500000
HIDDEN_DIM = 256
WANDB_PROJECT = "varshare-requeue-test"

SLURM_TIME = "00:10:00"
SIGNAL_SPEC = "B:USR1@120"  # Signal bash at 8 min; bash forwards to Python via trap


def submit_all():
    user = os.environ.get("USER", "nertinger")
    base_analysis = f"/netscratch/{user}/varshare/analysis/requeue_test"
    base_logs = f"/netscratch/{user}/varshare/logs/requeue_test"
    
    # Clean up old checkpoints and logs from previous test attempts
    print("Cleaning up old requeue test data...")
    for name in METHODS:
        old_analysis = os.path.join(base_analysis, name)
        if os.path.exists(old_analysis):
            shutil.rmtree(old_analysis)
            print(f"  Removed: {old_analysis}")
        old_logs = os.path.join(base_logs, name)
        if os.path.exists(old_logs):
            shutil.rmtree(old_logs)
            print(f"  Removed: {old_logs}")
    
    print()
    print("=" * 60)
    print("  SLURM REQUEUE TEST SUBMISSION (v4)")
    print(f"  Time limit: {SLURM_TIME} | Signal: USR1 at 8 min")
    print("  CPU-only (batch partition) for fast scheduling")
    print(f"  Total steps: {TOTAL_STEPS:,}")
    print("  Signal routing: B:USR1 -> bash trap -> Python child")
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
        
        # The key insight: SLURM's B: prefix sends the signal to bash only.
        # We must use a bash trap to forward it to the Python child process.
        # The Python process is launched in the background (&) so we can
        # catch signals in the parent bash process and forward them.
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

# Forward SIGUSR1 from bash to the Python child process
trap 'echo "[BASH] Caught SIGUSR1, forwarding to Python (PID=$CHILD_PID)..."; kill -USR1 $CHILD_PID' USR1

# Launch Python in background so bash can catch signals
{full_cmd} &
CHILD_PID=$!
echo "[BASH] Python child PID: $CHILD_PID"

# Wait for Python to finish (wait is interruptible by signals)
wait $CHILD_PID
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
    print(f"  0:00 - 4:00  : MetaWorld environment initialization")
    print(f"  4:00 - 8:00  : Training begins, progress lines appear in .out")
    print(f"  8:00         : SLURM sends SIGUSR1 to bash")
    print(f"  8:00         : Bash trap forwards SIGUSR1 to Python child")
    print(f"  8:00 - 8:05  : Python catches signal, finishes update, saves checkpoint")
    print(f"  8:05         : Python exits with code 99, bash requeues the job")
    print(f"  ~8:10        : Job restarts, loads checkpoint, resumes W&B run")
    print(f"\nVerification (after ~15 minutes):")
    print(f"  cat {base_logs}/det_base/*.out")
    print(f"  Look for: [BASH] Caught SIGUSR1 -> [WARNING] Caught SIGUSR1 -> [CHECKPOINT] -> [REQUEUE] -> [RESUME]")
    print(f"\n  Then check the verification report:")
    print(f"  find {base_analysis} -name 'requeue_verification.txt' -exec cat {{}} \\;")

if __name__ == "__main__":
    submit_all()
