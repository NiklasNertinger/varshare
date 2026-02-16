#!/bin/bash
# test_restart_baselines_v3.sh
# 
# Usage: ./scripts/test_restart_baselines_v3.sh
# Description:
#   1. Cleans up SCALED V3 HPO data (leaving Mega HPO V3 untouched).
#   2. Launches a SHORT test run (2 Trials, 10k Steps, MT4).

echo "==================================================="
echo "  TESTING V3 BASELINE RESTART (2 Trials, 10k Steps)"
echo "==================================================="
echo "WARNING: This will delete existing SCALED V3 HPO data."
echo "Mega HPO V3 data will be PRESERVED."
echo "Waiting 3 seconds..."
sleep 3

# 1. Clean up SCALED V3 databases & logs
echo "[1/3] Deleting Scaled V3 HPO database..."
rm -f /netscratch/$USER/varshare/analysis/scaled_v3/optuna_journal_scaled_v3.log
rm -f /netscratch/$USER/varshare/analysis/scaled_v3/optuna_journal_scaled_v3.log.lock

echo "[1b/3] Cleaning up Scaled V3 log files..."
rm -rf logs/hpo_std_v3/*
mkdir -p logs/hpo_std_v3

echo "[1c/3] Cleaning Scaled V3 Analysis Directory..."
export HPO_SCALED_DIR="/netscratch/$USER/varshare/analysis/scaled_v3"
rm -rf ${HPO_SCALED_DIR}
mkdir -p ${HPO_SCALED_DIR}

# 2. Configure Test Environment
echo "[2/3] Exporting Test Environment Variables..."
export HPO_N_TRIALS=2
export HPO_TIME_STEPS=30000
export HPO_TIME_LIMIT=00:30:00 # 30 mins
export HPO_EVAL_FREQ=5000
export HPO_MT_SETTING="MT4"

# 3. Submit Jobs
echo "[3/3] Submitting Baselines V3 (Test Mode)..."
bash scripts/submit_restart_baselines_v3.sh

echo "==================================================="
echo "  TEST LAUNCH COMPLETE"
echo "==================================================="
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/hpo_std_v3/shared_0.out"
