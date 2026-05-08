#!/bin/bash
#SBATCH --job-name=varshare-final-benchmark-v2
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/netscratch/%u/varshare/logs/final_benchmark_v2_%j.out
#SBATCH --error=/netscratch/%u/varshare/logs/final_benchmark_v2_%j.err

# ==============================================================================
# VarShare Final Evaluation Benchmarking Trigger Script (v2 Campaign)
# Runs champion selection across v2 HPOs, then launches the 15M step evaluations.
# Executed automatically after all v2 HPO worker array jobs have completed.
# ==============================================================================

echo "=========================================================================="
echo "LAUNCHING AUTOMATED CHAMPION BENCHMARKING CAMPAIGN (v2)"
echo "=========================================================================="

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
cd ~/varshare

mkdir -p /netscratch/$USER/varshare/logs

echo -e "\n[Step 1] Running HPO v2 Champion Hyperparameter Extraction..."
python scripts/extract_hpo_results_v2.py

echo -e "\n[Step 2] Launching 15,000,000-Step, 2-Seed Benchmarking Runs (v2)..."
python scripts/submit_final_campaign_v2.py --envs MT10

echo -e "\n=========================================================================="
echo "BENCHMARKING DISPATCH WORKFLOW COMPLETE (v2)"
echo "=========================================================================="
