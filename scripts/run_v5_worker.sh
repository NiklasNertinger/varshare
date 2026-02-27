#!/bin/bash
#SBATCH --job-name=v5_worker
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=/netscratch/%u/varshare/logs/v5_full_%A_%a.out

ENV_KEY=$1
METHOD=$2
ANALYSIS_DIR=/netscratch/$USER/varshare/analysis/det_hpo/v5_full/${ENV_KEY}/${METHOD}

mkdir -p $ANALYSIS_DIR

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

python scripts/optimize_v5.py \
    --env-key $ENV_KEY \
    --method $METHOD \
    --n-trials 1 \
    --storage $3 \
    --analysis-dir $ANALYSIS_DIR
