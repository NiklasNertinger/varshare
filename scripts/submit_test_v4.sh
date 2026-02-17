#!/bin/bash
# submit_test_v4.sh
# Usage: ./submit_test_v4.sh [OPTIONAL: env_key method]

# --- Configuration ---
# Test Settings
EPOCH_STEPS=10000
EVAL_FREQ=2000
N_TRIALS=1

# Paths
HPO_ANALYSIS_DIR="/netscratch/$USER/varshare/analysis/mega_v4_test"

# Slurm Settings
PARTITION="batch"
TIME="00:30:00" # Short time for test
CPUS="4"
MEM="4G"
ARRAY_SIZE="2" # Array 0-1 (2 concurrent jobs)

# Definitions
ENVS=("CP" "LL" "MT4" "IdenticalCP" "IdenticalLL")
# Test a subset of methods to verify pipeline
METHODS=("varshare_standard" "shared_embedding")

submit_test_job() {
    ENV_KEY=$1
    METHOD=$2
    CONSISTENT=$3
    
    # Construct Study Name
    STUDY_NAME="test_v4_${ENV_KEY}_${METHOD}"
    if [ "$CONSISTENT" == "true" ]; then
        STUDY_NAME="${STUDY_NAME}_consistent"
    fi
    
    # Separate DB per Environment (Test DBs)
    # Using JournalStorage (.log) to avoid SQLite NFS locking issues
    STORAGE_PATH="/netscratch/$USER/varshare/analysis/v4_test_${ENV_KEY}.log"
    
    echo "Submitting TEST: $STUDY_NAME (Env: $ENV_KEY, Method: $METHOD, DB: $STORAGE_PATH)"
    
    mkdir -p logs/hpo_test_v4
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${STUDY_NAME}
#SBATCH --output=logs/hpo_test_v4/${STUDY_NAME}_%a.out
#SBATCH --error=logs/hpo_test_v4/${STUDY_NAME}_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-$(($ARRAY_SIZE - 1))

# Activate Environment
source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare
export HPO_ANALYSIS_DIR=${HPO_ANALYSIS_DIR}
export HPO_TIME_STEPS=${EPOCH_STEPS}
export HPO_EVAL_FREQ=${EVAL_FREQ}

# Run Optimization
python scripts/optimize_v4.py \\
    --env-key "${ENV_KEY}" \\
    --method "${METHOD}" \\
    --consistent-noise "${CONSISTENT}" \\
    --storage "${STORAGE_PATH}" \\
    --n-trials ${N_TRIALS} 
EOT
}

# --- Submission Logic ---

if [ -z "$1" ]; then
    # Submit ALL Envs with subset of logic
    for env in "${ENVS[@]}"; do
         # Test 1 Baseline
         submit_test_job "$env" "shared_embedding" "false"
         # Test 1 VarShare
         submit_test_job "$env" "varshare_standard" "true"
    done
else
    # Manual Submission
    submit_test_job "$1" "$2" "$3"
fi
