#!/bin/bash
#SBATCH --job-name=hpo_scaled
#SBATCH --output=/netscratch/%u/varshare/logs/%x_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=RTXA6000,L40S,batch,RTX3090,A100-40GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-50%10

# =============================================================================
# DFKI Pegasus HPO Launcher (SCALED: 256-dim, 2M steps)
# =============================================================================

# 1. Setup Environment
source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

# 2. Run Optimization
# We use the array index mainly for job management, 
# although optuna naturally handles parallel workers via the shared DB.
# Each job runs a subset of the total trials desired.
# e.g. If we want 100 trials total, and we have array=1-10, each should run ~10 trials.
# But simply running n_trials=10 per worker is easier.

echo "Starting HPO Worker $SLURM_ARRAY_TASK_ID on $(hostname)"

# Point to analysis dir on netscratch (shared storage for DB)
STORAGE_DIR="/netscratch/$USER/varshare/analysis"

python scripts/optimize_mt10_varshare_scaled.py \
    --n-trials 5 \
    --analysis-dir $STORAGE_DIR

echo "Worker $SLURM_ARRAY_TASK_ID finished."
