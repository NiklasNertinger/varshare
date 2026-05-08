#!/bin/bash
# ==============================================================================
# Script to submit all 4 recommended HPO sweeps (v2 Campaign) to SLURM.
# Natively schedules the final v2 champion benchmarking via dependency chaining.
# ==============================================================================

METHODS=(
    "shared"
    "pcgrad"
    "soft_mod"
    "routing"
)

JOB_IDS=()

echo "Submitting 4 recommended HPO v2 sweeps (4M steps, 80k eval intervals)..."
for METHOD in "${METHODS[@]}"; do
    SUB_OUT=$(sbatch slurm/hpo_worker_array_v2.sh "${METHOD}")
    # Extract Job ID from output e.g. "Submitted batch job 123456"
    JOB_ID=$(echo "${SUB_OUT}" | grep -o -E '[0-9]+')
    echo "  -> Submitted ${METHOD} HPO sweep with Job ID: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

# Format Job IDs as a comma-separated list for SLURM dependency matching
DEP_JOB_LIST=$(IFS=,; echo "${JOB_IDS[*]}")

echo -e "\nAll HPO v2 jobs successfully scheduled on SLURM."
echo "Setting up native dependency chaining (v2)..."

# Submit final benchmarking run to execute ONLY after all HPO v2 array tasks successfully complete
BENCH_SUB=$(sbatch --dependency=afterok:${DEP_JOB_LIST} slurm/run_final_benchmarking_v2.sh)
BENCH_ID=$(echo "${BENCH_SUB}" | grep -o -E '[0-9]+')

echo "--------------------------------------------------------------------------"
echo "Success! Final Benchmarking v2 scheduled with Job ID: ${BENCH_ID}"
echo "  -> Note: Job ${BENCH_ID} is currently Pending (Dependency)."
echo "  -> It consumes 0 cluster GPU/CPU resources or active job slots."
echo "  -> It will automatically trigger once all 4 HPO arrays are completed."
echo "--------------------------------------------------------------------------"
