#!/bin/bash
#SBATCH --job-name=varshare-hpo-v2
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/netscratch/%u/varshare/logs/hpo_v2_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/hpo_v2_%A_%a.err
#SBATCH --array=1-60

# ==============================================================================
# VarShare HPO Worker Array v2 — Elastic Single-Trial Worker Loop (4M HPO)
# Uses SQLite in WAL mode with robust connection locks.
# 
# Usage: sbatch slurm/hpo_worker_array_v2.sh <shared|pcgrad|soft_mod|routing>
# ==============================================================================

if [ -z "$1" ]; then
    echo "Error: Must specify algorithm method to tune."
    echo "Usage: sbatch slurm/hpo_worker_array_v2.sh <shared|pcgrad|soft_mod|routing>"
    exit 1
fi

METHOD=$1
DB_FILE="/netscratch/$USER/varshare/test_hpo_v2.db"
STORAGE_URI="sqlite:///${DB_FILE}"

echo "=========================================================================="
echo "LAUNCHING VARSHARE HPO v2 TASK: ${SLURM_ARRAY_TASK_ID} / 60"
echo "Method / Algo: ${METHOD}"
echo "Storage URI: ${STORAGE_URI}"
echo "=========================================================================="

source /netscratch/$USER/varshare/venv/bin/activate
cd ~/varshare

mkdir -p /netscratch/$USER/varshare/logs
mkdir -p $(dirname ${DB_FILE})

if [ ! -f "${DB_FILE}" ]; then
    echo "Creating SQLite database with WAL journal mode..."
    sqlite3 ${DB_FILE} "PRAGMA journal_mode=WAL;"
fi

python optimize_v2.py \
    --env-key MT10 \
    --method ${METHOD} \
    --n-trials 1 \
    --storage ${STORAGE_URI} \
    --seed 1 \
    --analysis-dir "/netscratch/$USER/varshare/analysis_hpo_v2"

echo "=========================================================================="
echo "HPO TASK ${SLURM_ARRAY_TASK_ID} COMPLETE!"
echo "=========================================================================="
