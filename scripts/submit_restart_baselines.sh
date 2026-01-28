#!/bin/bash
# Submit only the fixed baselines
# PCGrad, PaCo, SoftMod

# PCGrad
sbatch --job-name=hpo_pcgrad_scaled \
    --output=logs/hpo_pcgrad_scaled_%A_%a.out \
    --error=logs/hpo_pcgrad_scaled_%A_%a.err \
    --array=1-50 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_pcgrad_scaled.py --n-trials 50 --analysis-dir /netscratch/\$USER/varshare/analysis"

# PaCo
sbatch --job-name=hpo_paco_scaled \
    --output=logs/hpo_paco_scaled_%A_%a.out \
    --error=logs/hpo_paco_scaled_%A_%a.err \
    --array=1-50 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_paco_scaled.py --n-trials 50 --analysis-dir /netscratch/\$USER/varshare/analysis"

# SoftMod
sbatch --job-name=hpo_softmod_scaled \
    --output=logs/hpo_softmod_scaled_%A_%a.out \
    --error=logs/hpo_softmod_scaled_%A_%a.err \
    --array=1-50 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_soft_mod_scaled.py --n-trials 50 --analysis-dir /netscratch/\$USER/varshare/analysis"
