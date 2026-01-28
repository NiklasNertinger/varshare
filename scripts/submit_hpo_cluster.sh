#!/bin/bash
#SBATCH --job-name=hpo_mt10
#SBATCH --output=/netscratch/%u/varshare/logs/%x_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/%x_%A_%a.err
#SBATCH --time=24:00:00

# Partitions: Prioritize RTXA6000 (Highest Availability) > L40S > batch > RTX3090 (Busy)
#SBATCH --partition=RTXA6000,L40S,batch,RTX3090,A100-40GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8      # Bumped to 8 to match colleague's safe baseline
#SBATCH --mem=8G
#SBATCH --array=1-50%5

# Notifications (Optional - Uncomment and set email)
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@dfki.de

set -euo pipefail

# =============================================================================
# DFKI Pegasus HPO Launcher
# =============================================================================

# 1. Setup Environment
# 'module' command is proving problematic in non-interactive batch mode.
# We rely on the pre-installed drivers and the venv's PyTorch/CUDA binaries.
# Use ABSOLUTE path to venv to avoid "No such file" errors
source /netscratch/$USER/varshare/venv/bin/activate

# Add the project root to PYTHONPATH so scripts can find 'src'
# We use $HOME/varshare explicitly to be safe across nodes
export PYTHONPATH="${PYTHONPATH:-}:$HOME/varshare"

# 2. Database Location
# 2. Storage Setup (JournalStorage)
# We use JournalStorage (file-based) which is robust on NFS. 
# The log file will be created in RESULTS_DIR automatically by the python script.
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
# Use -u for unbuffered output so tail -f works immediately
srun python -u "$SCRIPT_NAME" --n-trials 1 --analysis-dir "$RESULTS_DIR"
