#!/bin/bash
# Submit ALL Standalone HPO Studies (Baselines + Scaled VarShare)
# Includes: VarShare, Shared, PCGrad, PaCo, SoftMod, Oracle

ARRAY_SIZE="${HPO_N_TRIALS:-50}"
echo "Submitting Baselines with Array Size: $ARRAY_SIZE"

# 1. VarShare Scaled (Original Study)
sbatch --job-name=hpo_varshare_scaled \
    --output=logs/hpo_std_varshare_%A_%a.out \
    --error=logs/hpo_std_varshare_%A_%a.err \
    --array=0-$(($ARRAY_SIZE - 1)) \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_varshare_scaled.py --n-trials $ARRAY_SIZE --analysis-dir /netscratch/\$USER/varshare/analysis"

# 2. Shared Baseline
sbatch --job-name=hpo_shared_scaled \
    --output=logs/hpo_std_shared_%A_%a.out \
    --error=logs/hpo_std_shared_%A_%a.err \
    --array=0-$(($ARRAY_SIZE - 1)) \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_shared_scaled.py --n-trials $ARRAY_SIZE --analysis-dir /netscratch/\$USER/varshare/analysis"

# 3. PCGrad Baseline
sbatch --job-name=hpo_pcgrad_scaled \
    --output=logs/hpo_std_pcgrad_%A_%a.out \
    --error=logs/hpo_std_pcgrad_%A_%a.err \
    --array=0-$(($ARRAY_SIZE - 1)) \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_pcgrad_scaled.py --n-trials $ARRAY_SIZE --analysis-dir /netscratch/\$USER/varshare/analysis"

# 4. PaCo Baseline
sbatch --job-name=hpo_paco_scaled \
    --output=logs/hpo_std_paco_%A_%a.out \
    --error=logs/hpo_std_paco_%A_%a.err \
    --array=0-$(($ARRAY_SIZE - 1)) \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_paco_scaled.py --n-trials $ARRAY_SIZE --analysis-dir /netscratch/\$USER/varshare/analysis"

# 5. SoftMod Baseline
sbatch --job-name=hpo_softmod_scaled \
    --output=logs/hpo_std_softmod_%A_%a.out \
    --error=logs/hpo_std_softmod_%A_%a.err \
    --array=0-$(($ARRAY_SIZE - 1)) \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_soft_mod_scaled.py --n-trials $ARRAY_SIZE --analysis-dir /netscratch/\$USER/varshare/analysis"

# 6. Oracle Baseline
sbatch --job-name=hpo_oracle_scaled \
    --output=logs/hpo_std_oracle_%A_%a.out \
    --error=logs/hpo_std_oracle_%A_%a.err \
    --array=0-$(($ARRAY_SIZE - 1)) \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="source /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; python scripts/optimize_mt10_oracle_scaled.py --n-trials $ARRAY_SIZE --analysis-dir /netscratch/\$USER/varshare/analysis"
