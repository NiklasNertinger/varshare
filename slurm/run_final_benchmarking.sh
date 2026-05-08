#!/bin/bash
#SBATCH --job-name=varshare-final-benchmark
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/netscratch/%u/varshare/logs/final_benchmark_%j.out
#SBATCH --error=/netscratch/%u/varshare/logs/final_benchmark_%j.err

# ==============================================================================
# VarShare Final Evaluation Benchmarking Trigger Script
# Runs champion selection across HPOs, then launches the 5M step evaluations.
# Executed automatically after all HPO worker array jobs have completed.
# ==============================================================================

echo "=========================================================================="
echo "LAUNCHING AUTOMATED CHAMPION BENCHMARKING CAMPAIGN"
echo "=========================================================================="

# Activate workspace virtual environment
source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
cd ~/varshare

# Ensure logs folder exists
mkdir -p /netscratch/$USER/varshare/logs

# 1. Extract best champion hyperparameters from SQLite database
echo -e "\n[Step 1] Running HPO Champion Hyperparameter Extraction..."
python scripts/extract_hpo_results.py

# 2. Trigger the final 5,000,000 step benchmarking evaluations for MT10
echo -e "\n[Step 2] Launching 5,000,000-Step, 2-Seed Benchmarking Runs..."
python scripts/submit_final_campaign.py --envs MT10

echo -e "\n=========================================================================="
echo "BENCHMARKING DISPATCH WORKFLOW COMPLETE"
echo "=========================================================================="
