#!/bin/bash
#SBATCH --job-name=master-overnight
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/netscratch/%u/varshare/logs/master_pipeline_%j.out
#SBATCH --error=/netscratch/%u/varshare/logs/master_pipeline_%j.err

# ==============================================================================
# VarShare Master Overnight Pipeline Submitter
# Submits the HPO studies, monitors them, selects champions, and launches
# the 5M step, 2-seed MT10 benchmarking trials automatically.
# 
# Usage: sbatch slurm/run_overnight_pipeline.sh
# ==============================================================================

echo "=========================================================================="
echo "LAUNCHING PERSISTENT MASTER OVERNIGHT ORCHESTRATOR"
echo "=========================================================================="

# Activate workspace virtual environment
source /netscratch/$USER/varshare/venv/bin/activate
cd ~/varshare

# Run the master daemon
python scripts/master_watcher.py

echo "=========================================================================="
echo "MASTER OVERNIGHT ORCHESTRATOR TERMINATED"
echo "=========================================================================="
