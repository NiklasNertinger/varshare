#!/bin/bash
# submit_restart_baselines_v3.sh
# Submit ALL Standalone HPO Studies (Baselines + Scaled VarShare) for V3

ARRAY_SIZE="${HPO_N_TRIALS:-30}"
export HPO_TIME_STEPS="${HPO_TIME_STEPS:-2500000}"
# Overide storage file for the python scripts (V3)
export HPO_STORAGE_FILE="optuna_journal_scaled_v3.log"
export HPO_ANALYSIS_DIR="${HPO_SCALED_DIR:-/netscratch/$USER/varshare/analysis/scaled_v3}"
export HPO_EVAL_FREQ="${HPO_EVAL_FREQ:-25000}"
export HPO_MT_SETTING="${HPO_MT_SETTING:-MT4}"

TIME_LIMIT="${HPO_TIME_LIMIT:-06:00:00}"

echo "Submitting Baselines V3 with: Trials=$ARRAY_SIZE, Steps=$HPO_TIME_STEPS, Freq=$HPO_EVAL_FREQ, Setting=$HPO_MT_SETTING"
mkdir -p logs/hpo_std_v3

# 1. VarShare Scaled
sbatch --job-name=hpo_varshare_v3 \
    --output=logs/hpo_std_v3/varshare_%a.out \
    --error=logs/hpo_std_v3/varshare_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; export HPO_EVAL_FREQ=$HPO_EVAL_FREQ; export HPO_MT_SETTING=$HPO_MT_SETTING; python scripts/optimize_mt10_varshare_scaled.py --n-trials 1 --analysis-dir $HPO_ANALYSIS_DIR"

# 2. Shared Baseline
sbatch --job-name=hpo_shared_v3 \
    --output=logs/hpo_std_v3/shared_%a.out \
    --error=logs/hpo_std_v3/shared_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; export HPO_EVAL_FREQ=$HPO_EVAL_FREQ; export HPO_MT_SETTING=$HPO_MT_SETTING; python scripts/optimize_mt10_shared_scaled.py --n-trials 1 --analysis-dir $HPO_ANALYSIS_DIR"

# 3. PCGrad Baseline
sbatch --job-name=hpo_pcgrad_v3 \
    --output=logs/hpo_std_v3/pcgrad_%a.out \
    --error=logs/hpo_std_v3/pcgrad_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; export HPO_EVAL_FREQ=$HPO_EVAL_FREQ; export HPO_MT_SETTING=$HPO_MT_SETTING; python scripts/optimize_mt10_pcgrad_scaled.py --n-trials 1 --analysis-dir $HPO_ANALYSIS_DIR"

# 4. PaCo Baseline
sbatch --job-name=hpo_paco_v3 \
    --output=logs/hpo_std_v3/paco_%a.out \
    --error=logs/hpo_std_v3/paco_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; export HPO_EVAL_FREQ=$HPO_EVAL_FREQ; export HPO_MT_SETTING=$HPO_MT_SETTING; python scripts/optimize_mt10_paco_scaled.py --n-trials 1 --analysis-dir $HPO_ANALYSIS_DIR"

# 5. SoftMod Baseline
sbatch --job-name=hpo_softmod_v3 \
    --output=logs/hpo_std_v3/softmod_%a.out \
    --error=logs/hpo_std_v3/softmod_%a.err \
    --array=0-$(($ARRAY_SIZE - 1))%5 \
    --partition=batch \
    --cpus-per-task=8 \
    --mem=8G \
    --time=${TIME_LIMIT} \
    --wrap=". /netscratch/\$USER/varshare/venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare; export HPO_STORAGE_FILE=$HPO_STORAGE_FILE; export HPO_TIME_STEPS=$HPO_TIME_STEPS; export HPO_EVAL_FREQ=$HPO_EVAL_FREQ; export HPO_MT_SETTING=$HPO_MT_SETTING; python scripts/optimize_mt10_soft_mod_scaled.py --n-trials 1 --analysis-dir $HPO_ANALYSIS_DIR"
