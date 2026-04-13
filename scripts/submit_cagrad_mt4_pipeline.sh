#!/bin/bash
#SBATCH --job-name=cagrad_mt4_orchestrator
#SBATCH --output=/netscratch/%u/varshare/logs/%x_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/%x_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --partition=RTXA6000,L40S,batch,RTX3090,A100-40GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# This master script runs the Python orchestrator which handles HPO locally 
# and then automatically spawns SLURM children for the full 10M tests.

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

echo "Starting CAGrad MT4 Orchestrator Pipeline on $(hostname)"
python scripts/run_cagrad_mt4_pipeline.py
echo "Orchestrator finished distributing SLURM workflows."
