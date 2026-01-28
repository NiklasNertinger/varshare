#!/bin/bash
#SBATCH --job-name=hpo_softmod_scaled
#SBATCH --output=/netscratch/%u/varshare/logs/%x_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=RTXA6000,L40S,batch,RTX3090,A100-40GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-50%5

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
STORAGE_DIR="/netscratch/$USER/varshare/analysis"

echo "Starting HPO Worker $SLURM_ARRAY_TASK_ID on $(hostname)"
python scripts/optimize_mt10_soft_mod_scaled.py --n-trials 5 --analysis-dir $STORAGE_DIR
echo "Worker $SLURM_ARRAY_TASK_ID finished."
