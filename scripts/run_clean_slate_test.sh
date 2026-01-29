#!/bin/bash
# run_clean_slate_test.sh
# 
# Usage: ./scripts/run_clean_slate_test.sh

echo "==================================================="
echo "  STARTING CLEAN SLATE SMOKE TEST (2 Trials, 60k Steps)"
echo "==================================================="

# 1. Abort Existing Runs
echo "[1/4] Aborting existing jobs..."
scancel -u $USER

# 2. Delete Databases (Scaled & Mega HPO only)
echo "[2/4] Deleting old HPO databases (keeping base journal)..."
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_mega.log
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_scaled.log
echo "[2b/4] Cleaning up log files..."
rm -rf logs/hpo_mega/*
rm -f logs/hpo_std_*

echo "[2c/4] Stopping Dashboard (to release DB lock)..."
pkill -f launch_dashboard.py || echo "Dashboard not running."

# 3. Configure Test Environment
echo "[3/4] Exporting Test Environment Variables..."
export HPO_N_TRIALS=2
export HPO_TIME_STEPS=60000
export HPO_TIME_LIMIT=04:00:00
export HPO_EVAL_FREQ=10000
# Also export batch-size related things? 
# n_steps is per rollout. 256 default is fine.
# 60k steps / 256 approx 234 updates.

# 4. Launch ALL Studies
echo "[4/4] Submitting Jobs..."

# A) Baselines
# Pass "hpo_restart_baselines" logic, but submit_restart_baselines.sh doesn't support generic args easily?
# It wraps "python scripts/optimize_*.py".
# optimize_*.py usually takes args.
# run_trial in hpo_utils.py READS os.environ.
# So simply running the submit scripts with these env vars set will work!

# Submit 11 variants (Mega HPO) + 2 new ones
# submit_mega_hpo.sh now includes all 13.
bash scripts/submit_mega_hpo.sh

# Submit 6 Baselines (Shared, PaCo, etc.)
# We need to make sure submit_restart_baselines.sh uses the same env vars?
# Slurm usually inherits env vars if --export=ALL (default) or if defined in script.
# Our scripts manually `source venv` inside `sbatch`.
# Sbatch inherits environment variables from the submission shell unless --export=NONE is used.
# So `export HPO_N_TRIALS=2` here translates to the job environment!
bash scripts/submit_restart_baselines.sh

echo "==================================================="
echo "  ALL JOBS SUBMITTED for SMOKE TEST"
echo "  Monitor with: squeue -u $USER"
echo "==================================================="
