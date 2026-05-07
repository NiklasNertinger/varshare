#!/bin/bash
# ==============================================================================
# Script to submit all 9 HPO sweeps to SLURM
# Outputs the Job IDs so that the watcher daemon can track them.
# ==============================================================================

METHODS=(
    "shared"
    "independent"
    "pcgrad"
    "cagrad"
    "paco"
    "soft_mod"
    "moore"
    "base"
    "routing"
)

JOB_IDS=()

echo "Submitting 9 HPO sweeps to SLURM..."
for METHOD in "${METHODS[@]}"; do
    SUB_OUT=$(sbatch slurm/hpo_worker_array.sh "${METHOD}")
    # Extract Job ID from output e.g. "Submitted batch job 123456"
    JOB_ID=$(echo "${SUB_OUT}" | grep -o -E '[0-9]+')
    echo "  -> Submitted ${METHOD} HPO sweep with Job ID: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

# Output space-separated Job IDs to stdout for the python script to parse
echo "ALL_JOB_IDS: ${JOB_IDS[*]}"
