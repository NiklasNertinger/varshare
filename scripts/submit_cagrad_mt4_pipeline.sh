#!/bin/bash
# submit_cagrad_mt4_pipeline.sh
# This script completely automates the HPO arrays and subsequent Full Evals using Slurm dependencies.

LOG_PATH="/netscratch/$USER/varshare/logs"
mkdir -p $LOG_PATH

# 1. Start Baseline HPO Array
JOB1=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=hpo_base_cagrad
#SBATCH --output=${LOG_PATH}/%x_%A_%a.out
#SBATCH --error=${LOG_PATH}/%x_%A_%a.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --array=1-15%5

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare
python scripts/run_cagrad_mt4_pipeline.py --mode hpo_base
EOT
)
echo "Submitted HPO Base CAGrad array: $JOB1"

# 2. Start Deterministic HPO Array
JOB2=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=hpo_det_cagrad
#SBATCH --output=${LOG_PATH}/%x_%A_%a.out
#SBATCH --error=${LOG_PATH}/%x_%A_%a.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --array=1-15%5

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare
python scripts/run_cagrad_mt4_pipeline.py --mode hpo_det
EOT
)
echo "Submitted HPO Det CAGrad array: $JOB2"

# 3. Schedule final evaluations (will only run when Job1 and Job2 succeed entirely)
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=cagrad_launch_evals
#SBATCH --output=${LOG_PATH}/%x_%A.out
#SBATCH --error=${LOG_PATH}/%x_%A.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --dependency=afterok:${JOB1}:${JOB2}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare
python scripts/run_cagrad_mt4_pipeline.py --mode full_evals
EOT

echo "Submitted final evaluator orchestrator to wait for completion."
