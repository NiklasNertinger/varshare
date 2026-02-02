#!/bin/bash
# run_production_v2.sh
# 
# Usage: ./scripts/run_production_v2.sh
# Description:
#   1. Aborts all existing jobs.
#   2. Cleans up ONLY V2 HPO databases and logs (Preserves V1).
#   3. Launches the V2 PRODUCTION run (30 Trials, 10M Steps, No Pruning).

echo "==================================================="
echo "  STARTING V2 PRODUCTION RUN (30 Trials, 10M Steps)"
echo "==================================================="
echo "WARNING: This will delete existing V2 HPO data (optuna_journal_*_v2.log)"
echo "Waiting 5 seconds... CTRL+C to cancel."
sleep 5

# 1. Abort Existing Runs
echo "[1/4] Aborting existing jobs..."
scancel -u $USER

# 2. Delete V2 Databases & Logs (Safe for V1)
echo "[2/4] Deleting old V2 HPO databases..."
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_mega_v2.log
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_scaled_v2.log

echo "[2b/4] Cleaning up V2 log files..."
rm -rf logs/hpo_mega_v2/*
rm -rf logs/hpo_std_v2/*

echo "[2c/4] Stopping Dashboard..."
pkill -f launch_dashboard.py || echo "Dashboard not running."

# 3. Configure Production Environment
echo "[3/4] Exporting V2 Environment Variables..."
export HPO_N_TRIALS=2 # TEST MODE
export HPO_TIME_STEPS=15000 # TEST MODE
export HPO_TIME_LIMIT=04:00:00 # Safe limit
export HPO_EVAL_FREQ=50000
export HPO_ANALYSIS_DIR="analysis/mega_v2"

echo "[3b/4] Cleaning V2 Analysis Directory..."
rm -rf ${HPO_ANALYSIS_DIR}
mkdir -p ${HPO_ANALYSIS_DIR}

# 4. Submit Jobs
echo "[4/4] Submitting Jobs..."

echo ">>> Submitting Baselines V2 & Scaled VarShare..."
bash scripts/submit_restart_baselines_v2.sh

echo ">>> Submitting Mega HPO V2..."
bash scripts/submit_mega_hpo_v2.sh

echo "==================================================="
echo "  V2 LAUNCH COMPLETE"
echo "==================================================="
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/hpo_mega_v2/mt10_varshare_base_0.out"
