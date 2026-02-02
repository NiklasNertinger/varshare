#!/bin/bash
# submit_restart_baselines_v2.sh
# Submit ALL Standalone HPO Studies (Baselines + Scaled VarShare) for V2

ARRAY_SIZE="${HPO_N_TRIALS:-30}"
export HPO_TIME_STEPS="${HPO_TIME_STEPS:-10000000}"
# Overide storage file for the python scripts
export HPO_STORAGE_FILE="optuna_journal_scaled_v2.log" 
TIME_LIMIT="${HPO_TIME_LIMIT:-12:00:00}"

echo "Submitting Baselines V2 with Array Size: $ARRAY_SIZE, Steps: $HPO_TIME_STEPS"
mkdir -p logs/hpo_std_v2

# 1. VarShare Scaled
sbatch --job-name=hpo_varshare_v2 \
    --output=logs/hpo_std_v2/varshare_%a.out \
    --error=logs/hpo_std_v2/varshare_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; python scripts/optimize_mt10_varshare_scaled.py --n-trials 1 --analysis-dir /netscratch/\$USER/varshare/analysis"

# 2. Shared Baseline
sbatch --job-name=hpo_shared_v2 \
    --output=logs/hpo_std_v2/shared_%a.out \
    --error=logs/hpo_std_v2/shared_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; python scripts/optimize_mt10_shared_scaled.py --n-trials 1 --analysis-dir /netscratch/\$USER/varshare/analysis"

# 3. PCGrad Baseline
sbatch --job-name=hpo_pcgrad_v2 \
    --output=logs/hpo_std_v2/pcgrad_%a.out \
    --error=logs/hpo_std_v2/pcgrad_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; python scripts/optimize_mt10_pcgrad_scaled.py --n-trials 1 --analysis-dir /netscratch/\$USER/varshare/analysis"

# 4. PaCo Baseline
sbatch --job-name=hpo_paco_v2 \
    --output=logs/hpo_std_v2/paco_%a.out \
    --error=logs/hpo_std_v2/paco_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; python scripts/optimize_mt10_paco_scaled.py --n-trials 1 --analysis-dir /netscratch/\$USER/varshare/analysis"

# 5. SoftMod Baseline
sbatch --job-name=hpo_softmod_v2 \
    --output=logs/hpo_std_v2/softmod_%a.out \
    --error=logs/hpo_std_v2/softmod_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; python scripts/optimize_mt10_soft_mod_scaled.py --n-trials 1 --analysis-dir /netscratch/\$USER/varshare/analysis"

