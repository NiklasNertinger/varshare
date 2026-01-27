#!/bin/bash
#SBATCH --job-name=hpo_mt10
#SBATCH --output=/netscratch/%u/varshare/logs/%x_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/%x_%A_%a.err
#SBATCH --time=24:00:00

# Partitions: Try multiple to reduce queue time (Priority order)
#SBATCH --partition=RTX3090,RTXA6000,L40S,A100-40GB 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8      # Bumped to 8 to match colleague's safe baseline
#SBATCH --mem=32G
#SBATCH --array=1-50%5

# Notifications (Optional - Uncomment and set email)
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@dfki.de

set -euo pipefail

# =============================================================================
# DFKI Pegasus HPO Launcher
# =============================================================================

# 1. Setup Environment
module load cuda/12.1 
source .venv/bin/activate

# 2. Database Location
DB_DIR="/netscratch/$USER/varshare/hpo_db"
mkdir -p "$DB_DIR"
DB_PATH="$DB_DIR/optuna_mt10.db"
ln -sf "$DB_PATH" optuna_mt10.db

# 3. Select Algorithm
ALGO=$1
if [ -z "$ALGO" ]; then
    echo "Error: No algorithm specified. Usage: sbatch scripts/submit_hpo_cluster.sh <algo>"
    exit 1
fi

SCRIPT_NAME="scripts/optimize_mt10_${ALGO}.py"
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Script $SCRIPT_NAME not found."
    exit 1
fi

echo "Job: $SLURM_JOB_ID | Task: $SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME"

# RESULTS_DIR
RESULTS_DIR="/netscratch/$USER/varshare/analysis"
mkdir -p "$RESULTS_DIR"

# 4. Run Optimization with srun (Better process tracking)
# Python runs ONE trial (--n-trials 1)
srun python "$SCRIPT_NAME" --n-trials 1 --analysis-dir "$RESULTS_DIR"
