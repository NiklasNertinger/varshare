#!/bin/bash
# run_production_full.sh
# 
# Usage: ./scripts/run_production_full.sh
# Description:
#   1. Aborts all existing jobs.
#   2. Cleans up old HPO databases and logs (DANGEROUS: DELETES DATA).
#   3. Launches the FULL production run (50 Trials, 2.5M Steps).

echo "==================================================="
echo "  STARTING FULL PRODUCTION RUN (50 Trials, 2.5M Steps)"
echo "==================================================="
echo "WARNING: This will delete existing HPO databases /netscratch/$USER/varshare/analysis/optuna_journal_*.log"
echo "Waiting 5 seconds... CTRL+C to cancel."
sleep 5

# 1. Abort Existing Runs
echo "[1/4] Aborting existing jobs..."
scancel -u $USER

# 2. Delete Databases & Logs
echo "[2/4] Deleting old HPO databases..."
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_mega.log
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_scaled.log

echo "[2b/4] Cleaning up log files..."
rm -rf logs/hpo_mega/*
rm -f logs/hpo_std_*

echo "[2c/4] Stopping Dashboard (to release DB lock)..."
pkill -f launch_dashboard.py || echo "Dashboard not running."

# 3. Configure Production Environment
echo "[3/4] Exporting Production Environment Variables..."
export HPO_N_TRIALS=50
export HPO_TIME_STEPS=2500000
export HPO_TIME_LIMIT=06:00:00
export HPO_EVAL_FREQ=50000

# 4. Submit Jobs
echo "[4/4] Submitting Jobs..."

echo ">>> Submitting Baselines & Scaled VarShare..."
bash scripts/submit_restart_baselines.sh

echo ">>> Submitting Mega HPO Variants..."
bash scripts/submit_mega_hpo.sh

echo "==================================================="
echo "  PRODUCTION LAUNCH COMPLETE"
echo "==================================================="
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/hpo_mega/mt10_varshare_base_0.out"
