#!/bin/bash
# ==============================================================================
# Script to submit all 9 HPO sweeps to SLURM and chain the final benchmark.
# Natively schedules champion evaluation via SLURM dependency chaining.
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

echo "Submitting 9 HPO sweeps to SLURM (each with 60 single-trial worker slots)..."
for METHOD in "${METHODS[@]}"; do
    SUB_OUT=$(sbatch slurm/hpo_worker_array.sh "${METHOD}")
    # Extract Job ID from output e.g. "Submitted batch job 123456"
    JOB_ID=$(echo "${SUB_OUT}" | grep -o -E '[0-9]+')
    echo "  -> Submitted ${METHOD} HPO sweep with Job ID: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

# Format Job IDs as a comma-separated list for SLURM dependency matching
DEP_JOB_LIST=$(IFS=,; echo "${JOB_IDS[*]}")

echo -e "\nAll HPO jobs successfully scheduled on SLURM."
echo "Setting up native dependency chaining..."

# Submit final benchmarking run to execute ONLY after all HPO array tasks successfully complete (afterok)
BENCH_SUB=$(sbatch --dependency=afterok:${DEP_JOB_LIST} slurm/run_final_benchmarking.sh)
BENCH_ID=$(echo "${BENCH_SUB}" | grep -o -E '[0-9]+')

echo "--------------------------------------------------------------------------"
echo "Success! Final Benchmarking scheduled with Job ID: ${BENCH_ID}"
echo "  -> Note: Job ${BENCH_ID} is currently Pending (Dependency)."
echo "  -> It consumes 0 cluster GPU/CPU resources or active job slots."
echo "  -> It will automatically trigger once all 9 HPO arrays are completed."
echo "--------------------------------------------------------------------------"
