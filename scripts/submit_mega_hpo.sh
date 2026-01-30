#!/bin/bash
# submit_mega_hpo.sh
# Usage: ./submit_mega_hpo.sh [OPTIONAL: study_name]
# If study_name is provided, submits ONLY that study.
# If no argument, submits ALL 11 studies.

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

# Common Slurm Settings
PARTITION="batch" # Or whatever partition was used before
TIME="4:00:00" # 4 hours per trial (2M steps on CPU takes ~2-3h usually)
CPUS="8" # 8 CPUs per trial
MEM="8G" # 8GB RAM per trial
ARRAY_SIZE="${HPO_N_TRIALS:-50}" # Default 50, override with env var
export HPO_TIME_STEPS="${HPO_TIME_STEPS:-2500000}"

submit_study() {
    STUDY_NAME=$1
    echo "Submitting HPO for: $STUDY_NAME"
    
    # Create log dir
    mkdir -p logs/hpo_mega
    
    # Define Storage (Must be defined here for heredoc expansion)
    STORAGE_PATH="/netscratch/$USER/varshare/analysis/optuna_journal_mega.log"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${STUDY_NAME}
#SBATCH --output=logs/hpo_mega/${STUDY_NAME}_%a.out
#SBATCH --error=logs/hpo_mega/${STUDY_NAME}_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-$(($ARRAY_SIZE - 1))%5

# 1. Setup Environment
source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

# 2. Storage is expanded from parent shell
# STORAGE_PATH is ${STORAGE_PATH}

# Execute Specific Optimizer Script
echo "Running Study: ${STUDY_NAME}"
echo "Storage: ${STORAGE_PATH}"
python scripts/optimize_${STUDY_NAME}.py --storage-path "${STORAGE_PATH}" --n-trials 1

EOT
}

if [ -z "$1" ]; then
    echo "Submitting ALL 11 Studies..."
    for s in "${STUDIES[@]}"; do
        submit_study $s
    done
else
    # Check if valid study
    FOUND=0
    for s in "${STUDIES[@]}"; do
        if [ "$s" == "$1" ]; then
            FOUND=1
            break
        fi
    done
    
    if [ $FOUND -eq 1 ]; then
        submit_study $1
    else
        echo "Error: Unknown study '$1'. Available studies:"
        printf '%s\n' "${STUDIES[@]}"
        exit 1
    fi
fi
