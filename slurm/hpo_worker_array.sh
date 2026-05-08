#!/bin/bash
#SBATCH --job-name=varshare-hpo
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/netscratch/%u/varshare/logs/hpo_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/hpo_%A_%a.err
#SBATCH --array=1-60
# ==============================================================================
# VarShare HPO Worker Array — 6 Parallel Workers, 60 Trials Total (10 per worker)
# Uses SQLite in WAL (Write-Ahead Logging) mode with robust connection locks.
# 
# Usage: sbatch slurm/hpo_worker_array.sh <ALGO>
# Examples:
#   sbatch slurm/hpo_worker_array.sh shared
#   sbatch slurm/hpo_worker_array.sh routing
# ==============================================================================

# Input validation
if [ -z "$1" ]; then
    echo "Error: Must specify algorithm method to tune."
    echo "Usage: sbatch slurm/hpo_worker_array.sh <shared|independent|pcgrad|cagrad|paco|soft_mod|moore|base|routing>"
    exit 1
fi

METHOD=$1
DB_FILE="/netscratch/$USER/varshare/test_hpo.db"
STORAGE_URI="sqlite:///${DB_FILE}"

echo "=========================================================================="
echo "LAUNCHING VARSHARE HPO TASK: ${SLURM_ARRAY_TASK_ID} / 6"
echo "Method / Algo: ${METHOD}"
echo "Storage URI: ${STORAGE_URI}"
echo "=========================================================================="

# Activate workspace virtual environment
source /netscratch/$USER/varshare/venv/bin/activate
cd ~/varshare

# Ensure logs and target folder structures exist
mkdir -p /netscratch/$USER/varshare/logs
mkdir -p $(dirname ${DB_FILE})

# Ensure SQLite optimization is run on first run
if [ ! -f "${DB_FILE}" ]; then
    echo "Creating SQLite database with WAL journal mode..."
    sqlite3 ${DB_FILE} "PRAGMA journal_mode=WAL;"
fi

# Run the Optuna Sweep Optimizer
# Each parallel array task index runs exactly 1 trial and exits immediately.
# SLURM automatically rotates new trials through active slots up to 60 total trials!
python optimize.py \
    --env-key MT10 \
    --method ${METHOD} \
    --n-trials 1 \
    --storage ${STORAGE_URI} \
    --seed 1 \
    --analysis-dir "/netscratch/$USER/varshare/analysis_hpo"

echo "=========================================================================="
echo "HPO TASK ${SLURM_ARRAY_TASK_ID} COMPLETE!"
echo "=========================================================================="
