#!/bin/bash
# submit_mini_test.sh
# Usage: ./submit_mini_test.sh

PARTITION="batch"
TIME="00:15:00"
CPUS="4"
MEM="8G" # Scaled up to 8G to safely accommodate MT10

mkdir -p /netscratch/$USER/varshare/logs/mini_test

VARIANTS=(
    "det_base"
    "soft_mod"
    "legacy_prob_base_consistent"
)

SCRIPTS=(
    "deterministic/scripts/train_det_ppo.py --variant base --max-grad-norm 1.0"
    "scripts/train_baseline_ppo.py --algo soft_mod --num-modules 4 --max-grad-norm 1.0"
    "scripts/train_varshare_ppo.py --variant standard --consistent-noise true --kl-beta 0.00043 --lr-actor 0.000117 --lr-critic 0.000259"
)

ENVS=("CP" "LL" "MT4" "MT10")

for ENV in "${ENVS[@]}"; do
    if [ "$ENV" == "CP" ]; then
        ENV_ARGS="--env-type IdenticalCartPole --hidden-dim 64"
    elif [ "$ENV" == "LL" ]; then
        ENV_ARGS="--env-type MultiTaskLunarLander --hidden-dim 64"
    elif [ "$ENV" == "MT4" ]; then
        ENV_ARGS="--env-type metaworld --mt-setting MT4 --hidden-dim 256"
    elif [ "$ENV" == "MT10" ]; then
        ENV_ARGS="--env-type metaworld --mt-setting MT10 --hidden-dim 256"
    fi

    for i in "${!VARIANTS[@]}"; do
        VARIANT="${VARIANTS[$i]}"
        SCRIPT_BASE="${SCRIPTS[$i]}"
        
        # Override hidden-dim for legacy variant (should always be 64x64 backbone)
        if [ "$VARIANT" == "legacy_prob_base_consistent" ]; then
             ENV_ARGS_MOD=$(echo $ENV_ARGS | sed 's/--hidden-dim 256/--hidden-dim 64/')
             SCRIPT_ARGS="${SCRIPT_BASE} ${ENV_ARGS_MOD}"
        else
             SCRIPT_ARGS="${SCRIPT_BASE} ${ENV_ARGS}"
        fi
        
        echo "Submitting mini-test for $VARIANT on $ENV"
        
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=mini_${ENV}_${VARIANT}
#SBATCH --output=/netscratch/$USER/varshare/logs/mini_test/mini_${ENV}_${VARIANT}_%j.out
#SBATCH --error=/netscratch/$USER/varshare/logs/mini_test/mini_${ENV}_${VARIANT}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$HOME/varshare

python ${SCRIPT_ARGS} \\
    --exp-name mini_test_${ENV}_${VARIANT} \\
    --seed 1 \\
    --total-timesteps 5000 \\
    --n-steps 256 \\
    --batch-size 64 \\
    --num-envs 4 \\
    --eval-mode True \\
    --eval-freq 2500 \\
    --analysis-dir /netscratch/$USER/varshare/analysis/mini_test/${ENV}_${VARIANT}
EOT
    done
done

echo "Mini test submission (CP, LL, MT4, MT10) complete!"
echo "Use 'squeue -u \$USER' to check status."
