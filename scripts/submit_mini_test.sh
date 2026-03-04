#!/bin/bash
# submit_mini_test.sh
# Usage: ./submit_mini_test.sh

PARTITION="batch"
TIME="00:15:00"
CPUS="4"
MEM="4G"

mkdir -p /netscratch/$USER/varshare/logs/mini_test
mkdir -p /netscratch/$USER/varshare/analysis/mini_test

VARIANTS=(
    "det_base"
    "soft_mod"
    "legacy_prob_base_consistent"
)

SCRIPTS=(
    "deterministic/scripts/train_det_ppo.py --variant base"
    "scripts/train_baseline_ppo.py --algo soft_mod --num-modules 4"
    "scripts/train_varshare_ppo.py --variant standard --consistent-noise true --kl-beta 0.00043 --lr-actor 0.000117 --lr-critic 0.000259 --hidden-dim 64"
)

for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    SCRIPT_ARGS="${SCRIPTS[$i]}"
    
    echo "Submitting mini-test for $VARIANT"
    
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=mini_${VARIANT}
#SBATCH --output=/netscratch/$USER/varshare/logs/mini_test/mini_${VARIANT}_%j.out
#SBATCH --error=/netscratch/$USER/varshare/logs/mini_test/mini_${VARIANT}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare

python ${SCRIPT_ARGS} \\
    --exp-name mini_test_${VARIANT} \\
    --seed 1 \\
    --total-timesteps 5000 \\
    --n-steps 256 \\
    --batch-size 64 \\
    --num-envs 4 \\
    --eval-mode True \\
    --eval-freq 2500 \\
    --env-type IdenticalCartPole \\
    --analysis-dir /netscratch/$USER/varshare/analysis/mini_test
EOT
done

echo "Mini test submission complete. Use 'squeue -u \$USER' to check status."
