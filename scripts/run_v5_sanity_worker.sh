#!/bin/bash
#SBATCH --job-name=v5_sanity
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/netscratch/%u/varshare/logs/v5_sanity_%A_%a.out

ENV_KEY=$1
METHOD=$2
ANALYSIS_DIR=/netscratch/$USER/varshare/analysis/det_hpo/v5_sanity/${ENV_KEY}/${METHOD}

mkdir -p $ANALYSIS_DIR

source .venv/bin/activate
export PYTHONPATH=.

python scripts/optimize_v5.py \
    --env-key $ENV_KEY \
    --method $METHOD \
    --n-trials 1 \
    --storage sqlite:////netscratch/$USER/varshare/v5_sanity_${ENV_KEY}.db \
    --analysis-dir $ANALYSIS_DIR \
    --test
