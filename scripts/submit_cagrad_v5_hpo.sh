#!/bin/bash

# Ensure netscratch directories exist
mkdir -p /netscratch/$USER/varshare/logs
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_full/MT4

ENVS=("MT4")

METHODS=(
    "cagrad" "det_cagrad"
)

# Make the worker script executable
chmod +x scripts/run_v5_worker.sh

echo "============================================="
echo "Submitting CAGrad v5 HPO Campaign"
echo "============================================="

for ENV in "${ENVS[@]}"; do
    echo ">>> Environment: $ENV"
    for METHOD in "${METHODS[@]}"; do
        echo "  -> Submitting Array (30 trials, max 5 parallel) for: $METHOD"
        sbatch --job-name="v5_${ENV}_${METHOD}" \
               --array=1-30%5 \
               scripts/run_v5_worker.sh $ENV $METHOD "/netscratch/$USER/varshare/v5_hpo_${ENV}.log"
    done
done

echo "============================================="
echo "All 2 Studies (60 total trials) submitted for CAGrad natively."
echo "============================================="
