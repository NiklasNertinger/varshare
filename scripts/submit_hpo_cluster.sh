#SBATCH --job-name=hpo_mt10
#SBATCH --output=/netscratch/%u/varshare/logs/%x_%A_%a.out  # Log to netscratch to save HOME quota
#SBATCH --error=/netscratch/%u/varshare/logs/%x_%A_%a.err
#SBATCH --time=24:00:00

# TIP: To see logs in Pegasus Dashboard, run: touch ~/.logoptin
# See: https://pegasus.dfki.de/docs/slurm-cluster/logging/
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=RTX3090  # Default to RTX3090, change to A100-40GB if needed
#SBATCH --array=1-50%5       # Run 50 trials total, max 5 concurrently for better BO

# Notifications (Optional - Uncomment and set email)
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@dfki.de

# =============================================================================
# DFKI Pegasus HPO Launcher
# =============================================================================

# 1. Setup Environment
# Load necessary modules (adjust version if needed)
module load cuda/12.1 

# Activate Virtual Env
source .venv/bin/activate

# 2. Database Location (Crucial for Parallelism)
# We use /netscratch/$USER which is shared and has space.
DB_DIR="/netscratch/$USER/varshare/hpo_db"
mkdir -p "$DB_DIR"
DB_PATH="$DB_DIR/optuna_mt10.db"

# Update hpo_utils.py to point here? 
# Instead of editing code, we can pass it via env var if we supported it, 
# OR just symlink it locally so the python script finds it.

# Create a local symlink to the shared DB
ln -sf "$DB_PATH" optuna_mt10.db

# 3. Select Algorithm based on CLI argument
# Usage: sbatch scripts/submit_hpo_cluster.sh varshare
ALGO=$1

if [ -z "$ALGO" ]; then
    echo "Error: No algorithm specified."
    echo "Usage: sbatch scripts/submit_hpo_cluster.sh <varshare|shared|paco|...>"
    exit 1
fi

SCRIPT_NAME="scripts/optimize_mt10_${ALGO}.py"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Script $SCRIPT_NAME not found."
    exit 1
fi

echo "=========================================================="
echo "Starting HPO Trial for $ALGO"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "DB Path: $DB_PATH"
echo "=========================================================="

# RESULTS_DIR for logs and plots (Avoid $HOME quota issues)
RESULTS_DIR="/netscratch/$USER/varshare/analysis"
mkdir -p "$RESULTS_DIR"

# 4. Run Optimization
# The python script runs ONE trial per execution (--n-trials 1)
python "$SCRIPT_NAME" --n-trials 1 --analysis-dir "$RESULTS_DIR"
