#!/bin/bash

# Ensure netscratch directories exist
mkdir -p /netscratch/$USER/varshare/logs
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_full

# Specific environments to be arrayed
ENVS=("CP" "LL" "MT4" "MT10")

# Make the worker script executable
chmod +x scripts/run_v5_worker.sh

echo "============================================="
echo "Submitting Routing Pipeline (HPO -> Eval)"
echo "============================================="

for ENV in "${ENVS[@]}"; do
    echo ">>> Environment: $ENV"
    
    # 1. Start the 25-trial HPO Array
    # Capture the array job ID
    JOB_ID=$(sbatch --parsable --job-name="hpo_routing_${ENV}" \
           --array=1-25%8 \
           scripts/run_v5_worker.sh $ENV det_routing "/netscratch/$USER/varshare/v5_hpo_${ENV}.log")
           
    echo "  -> Submitted HPO Array: $JOB_ID"
    
    # 2. Schedule the extraction and final evaluation after HPO succeeds
    # Using an inline sbatch for the dependent orchestrator
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=routing_eval_orchestrator_${ENV}
#SBATCH --output=/netscratch/$USER/varshare/logs/%x_%A.out
#SBATCH --error=/netscratch/$USER/varshare/logs/%x_%A.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:${JOB_ID}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare

echo "Extracting optimal hyperparameters..."
# Make sure log databases are locally accessible for the extractor. 
cp /netscratch/$USER/varshare/v5_hpo_${ENV}.log context/HPOs/v5_hpo_${ENV}.log

# Extracts and writes to best_hparams.json
python scripts/extract_optimal_hparams.py

echo "Launching automated evaluations for det_routing..."
python scripts/submit_final_campaign.py --envs ${ENV} --algo-filter det_routing
EOT
    
    echo "  -> Submitted automated continuation orchestrator"
done

echo "============================================="
echo "All HPOs and Evaluation dependencies submitted successfully."
echo "Use 'squeue -u \$USER' and 'python scripts/monitor_runs.py' to monitor."
echo "============================================="
