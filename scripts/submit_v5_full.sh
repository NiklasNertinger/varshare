#!/bin/bash

# Ensure netscratch directories exist
mkdir -p /netscratch/$USER/varshare/logs
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_full/CP
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_full/LL
mkdir -p /netscratch/$USER/varshare/analysis/det_hpo/v5_full/MT4

ENVS=("CP" "LL" "MT4")

METHODS=(
    "det_base" "det_lora" "det_gated" "det_l1" "det_pcgrad" 
    "det_decay" "det_film" "det_hyperprior" "det_ara"
    "shared_embedding" "shared_embedding_pcgrad" 
    "paco" "soft_mod" "varshare_prior_opt"
)

# Make the worker script executable
chmod +x scripts/run_v5_worker.sh

echo "============================================="
echo "Submitting v5 Full HPO Campaign"
echo "============================================="

for ENV in "${ENVS[@]}"; do
    echo ">>> Environment: $ENV"
    for METHOD in "${METHODS[@]}"; do
        echo "  -> Submitting Array (30 trials, max 5 parallel) for: $METHOD"
        sbatch --job-name="v5_${ENV}_${METHOD}" \
               --array=1-30%5 \
               scripts/run_v5_worker.sh $ENV $METHOD
    done
done

echo "============================================="
echo "All 42 Studies (1,260 total trials) submitted."
echo "============================================="
