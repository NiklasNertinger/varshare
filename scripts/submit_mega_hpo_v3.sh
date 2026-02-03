#!/bin/bash
# submit_mega_hpo_v3.sh
# Usage: ./submit_mega_hpo_v3.sh [OPTIONAL: study_name]

# Define Studies (Same as V1/V2)
STUDIES=(
    "mt10_varshare_base"
    "mt10_varshare_emb_onehot"
    "mt10_varshare_emb_learned"
    "mt10_varshare_lora"
    "mt10_varshare_partial"
    "mt10_varshare_reptile"
    "mt10_varshare_scaled_down"
    "mt10_varshare_fixed_prior_001"
    "mt10_varshare_fixed_prior_01"
    "mt10_varshare_fixed_prior_0001"
    "mt10_varshare_annealing"
    "mt10_varshare_trigger"
    "mt10_varshare_emp_bayes"
    "mt10_varshare_base_400"
    "mt10_varshare_base_64"
)

# Slurm Settings
PARTITION="batch"
# Defaults defined here but overridden by exported env vars in run_production_v3.sh
TIME="${HPO_TIME_LIMIT:-06:00:00}" 
CPUS="8"
MEM="8G"
ARRAY_SIZE="${HPO_N_TRIALS:-30}" # V3 Default: 30
# Default to 2.5M steps (MT4)
export HPO_TIME_STEPS="${HPO_TIME_STEPS:-2500000}" 
export HPO_ANALYSIS_DIR="${HPO_ANALYSIS_DIR:-/netscratch/$USER/varshare/analysis/mega_v3}"

# V3 Storage Path
STORAGE_PATH="/netscratch/$USER/varshare/analysis/optuna_journal_mega_v3.log"

submit_study() {
    STUDY_NAME=$1
    echo "Submitting MEGA HPO V3 for: $STUDY_NAME"
    
    mkdir -p logs/hpo_mega_v3
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${STUDY_NAME}_v3
#SBATCH --output=logs/hpo_mega_v3/${STUDY_NAME}_%a.out
#SBATCH --error=logs/hpo_mega_v3/${STUDY_NAME}_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-$(($ARRAY_SIZE - 1))%5

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
export HPO_TIME_STEPS=${HPO_TIME_STEPS}
export HPO_ANALYSIS_DIR=${HPO_ANALYSIS_DIR}
export HPO_EVAL_FREQ=${HPO_EVAL_FREQ:-25000}
export HPO_MT_SETTING=${HPO_MT_SETTING:-MT4}

python scripts/optimize_${STUDY_NAME}.py --storage-path "${STORAGE_PATH}" --n-trials 1
EOT
}

if [ -z "$1" ]; then
    for study in "${STUDIES[@]}"; do
        submit_study "$study"
    done
else
    submit_study "$1"
fi
