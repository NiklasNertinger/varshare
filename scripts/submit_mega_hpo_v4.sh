#!/bin/bash
# submit_mega_hpo_v4.sh
# Usage: ./submit_mega_hpo_v4.sh [OPTIONAL: env_key method consistent_noise]

# Configuration
HPO_ANALYSIS_DIR="/netscratch/$USER/varshare/analysis/mega_v4"

# Slurm Settings
PARTITION="batch"
TIME="06:00:00"
CPUS="8"
MEM="8G"
ARRAY_SIZE="30"

# Ordered Environments (Requested: CP -> LL -> MT4 -> Identical Envs)
ENVS=("CP" "LL" "MT4" "IdenticalCP" "IdenticalLL")

VARSHARE_METHODS=("varshare_standard" "varshare_scaled" "varshare_bayes" "varshare_partial" "varshare_prior_opt")
BASELINE_METHODS=("soft_modularization" "paco" "shared_embedding" "shared_embedding_pcgrad")

submit_job() {
    ENV_KEY=$1
    METHOD=$2
    CONSISTENT=$3
    
    # Construct Study Name
    STUDY_NAME="v4_${ENV_KEY}_${METHOD}"
    if [ "$CONSISTENT" == "true" ]; then
        STUDY_NAME="${STUDY_NAME}_consistent"
    fi
    
    # Separate DB per Environment
    STORAGE_PATH="sqlite:////netscratch/$USER/varshare/analysis/v4_hpo_${ENV_KEY}.db"
    
    echo "Submitting: $STUDY_NAME (Env: $ENV_KEY, Method: $METHOD, Consistent: $CONSISTENT, DB: $STORAGE_PATH)"
    
    mkdir -p logs/hpo_mega_v4
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${STUDY_NAME}
#SBATCH --output=logs/hpo_mega_v4/${STUDY_NAME}_%a.out
#SBATCH --error=logs/hpo_mega_v4/${STUDY_NAME}_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-$(($ARRAY_SIZE - 1))%10

# Activate Environment
source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare
export HPO_ANALYSIS_DIR=${HPO_ANALYSIS_DIR}

# Run Optimization
python scripts/optimize_v4.py \\
    --env-key "${ENV_KEY}" \\
    --method "${METHOD}" \\
    --consistent-noise "${CONSISTENT}" \\
    --storage "${STORAGE_PATH}" \\
    --n-trials 1 
EOT
}

# --- Submission Logic ---

if [ -z "$1" ]; then
    # Full Campaign (Ordered)
    for env in "${ENVS[@]}"; do
        echo ">>> Scheduling Environment: $env"
        
        # Baselines
        for method in "${BASELINE_METHODS[@]}"; do
            submit_job "$env" "$method" "false"
        done
        
        # VarShare
        for method in "${VARSHARE_METHODS[@]}"; do
            submit_job "$env" "$method" "false"
            submit_job "$env" "$method" "true"
        done
    done
else
    # Manual Submission
    submit_job "$1" "$2" "$3"
fi
