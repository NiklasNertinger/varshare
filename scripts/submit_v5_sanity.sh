#!/bin/bash

# Ensure netscratch directories exist
mkdir -p /netscratch/$USER/varshare/logs
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_sanity/CP
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_sanity/LL
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_sanity/MT4

ENVS=("CP" "LL" "MT4")

# Pick a couple representative methods for sanity
METHODS=("det_base" "paco")

chmod +x scripts/run_v5_sanity_worker.sh

echo "============================================="
echo "Submitting v5 Sanity HPO"
echo "============================================="

for ENV in "${ENVS[@]}"; do
    echo ">>> Environment: $ENV"
    for METHOD in "${METHODS[@]}"; do
        echo "  -> Submitting Sanity Array (2 trials) for: $METHOD"
        sbatch --job-name="v5_${ENV}_${METHOD}" \
               --array=1-2 \
               scripts/run_v5_sanity_worker.sh $ENV $METHOD
    done
done

echo "============================================="
echo "Sanity Check Studies submitted."
echo "============================================="
