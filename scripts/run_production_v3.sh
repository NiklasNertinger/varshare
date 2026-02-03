#!/bin/bash
# run_production_v3.sh
# 
# Usage: ./scripts/run_production_v3.sh
# Description:
#   1. Aborts all existing jobs.
#   2. Cleans up ONLY V3 HPO databases and logs (Preserves V1 and V2).
#   3. Launches the V3 PRODUCTION run (30 Trials, 2.5M Steps, MT4, Eval 25k).

echo "==================================================="
echo "  STARTING V3 PRODUCTION RUN (30 Trials, 2.5M Steps, MT4)"
echo "==================================================="
echo "WARNING: This will delete existing V3 HPO data (optuna_journal_*_v3.log)"
echo "V1 and V2 data will be PRESERVED."
echo "Waiting 5 seconds... CTRL+C to cancel."
sleep 5

# 1. Abort Existing Runs
echo "[1/4] Aborting existing jobs..."
scancel -u $USER

# 2. Delete V3 Databases & Logs (Safe for V1/V2)
echo "[2/4] Deleting V3 HPO databases..."
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_mega_v3.log
rm -f /netscratch/$USER/varshare/analysis/optuna_journal_scaled_v3.log

echo "[2b/4] Cleaning up V3 log files..."
rm -rf logs/hpo_mega_v3/*
rm -rf logs/hpo_std_v3/*
# Ensure dirs exist
mkdir -p logs/hpo_mega_v3 logs/hpo_std_v3

echo "[2c/4] Stopping Dashboard..."
pkill -f launch_dashboard.py || echo "Dashboard not running."

# 3. Configure Production Environment
echo "[3/4] Exporting V3 Environment Variables..."
export HPO_N_TRIALS=30
export HPO_TIME_STEPS=2500000
export HPO_TIME_LIMIT=06:00:00 # Safe limit for 2.5M steps
export HPO_EVAL_FREQ=25000
export HPO_MT_SETTING="MT4"
export HPO_ANALYSIS_DIR="/netscratch/$USER/varshare/analysis/mega_v3"
export HPO_SCALED_DIR="/netscratch/$USER/varshare/analysis/scaled_v3"

echo "[3b/4] Cleaning V3 Analysis Directory..."
rm -rf ${HPO_ANALYSIS_DIR} ${HPO_SCALED_DIR}
mkdir -p ${HPO_ANALYSIS_DIR} ${HPO_SCALED_DIR}

# 4. Submit Jobs
echo "[4/4] Submitting Jobs..."

echo ">>> Submitting Baselines V3 (Scaled)..."
bash scripts/submit_restart_baselines_v3.sh

echo ">>> Submitting Mega HPO V3..."
bash scripts/submit_mega_hpo_v3.sh

echo "==================================================="
echo "  V3 LAUNCH COMPLETE"
echo "==================================================="
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/hpo_mega_v3/mt10_varshare_base_0.out"
