#!/bin/bash
# submit_deterministic_v4.sh
# Usage: ./scripts/submit_deterministic_v4.sh

# Configuration
HPO_ANALYSIS_DIR="/netscratch/$USER/varshare/analysis/mega_v4"

# Slurm Settings
PARTITION="batch"
TIME="06:00:00"
CPUS="8"
MEM="8G"
ARRAY_SIZE="30"

# Ordered Environments (Requested: MT4, LL, CP)
ENVS=("MT4" "LL" "CP")
METHOD="varshare_standard"

for env in "${ENVS[@]}"; do
    # Construct Study Name
    STUDY_NAME="v4_deter_${env}_${METHOD}"
    
    # Separate DB per Environment
    STORAGE_PATH="/netscratch/$USER/varshare/analysis/v4_hpo_deterministic_${env}.log"
    
    echo "Submitting: $STUDY_NAME (Env: $env, DB: $STORAGE_PATH)"
    
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
python scripts/optimize_deterministic_v4.py \
    --env-key "${env}" \
    --method "${METHOD}" \
    --storage "${STORAGE_PATH}" \
    --n-trials 1 
EOT
done
