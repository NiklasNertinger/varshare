#!/bin/bash
# Submit Oracle HPO Scaled

sbatch --job-name=hpo_oracle_scaled \
    --output=logs/hpo_oracle_scaled_%A_%a.out \
    --error=logs/hpo_oracle_scaled_%A_%a.err \
    --array=1-50 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="python scripts/optimize_mt10_oracle_scaled.py --n-trials 50"
