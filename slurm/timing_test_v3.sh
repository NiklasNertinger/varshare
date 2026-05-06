#!/bin/bash
#SBATCH --job-name=varshare-timing-v3
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/netscratch/%u/varshare/logs/timing_v3_%A_%a.out
#SBATCH --error=/netscratch/%u/varshare/logs/timing_v3_%A_%a.err
#SBATCH --array=1-12

# ============================================================
# VarShare Timing Test v3 — 12 runs, 150k steps each, num-envs=10
# Submit with: sbatch slurm/timing_test_v3.sh
# ============================================================

# Setup
source /netscratch/$USER/varshare/venv/bin/activate
cd ~/varshare
mkdir -p /netscratch/$USER/varshare/logs

STEPS=150000
NUM_ENVS=10
EVAL_FREQ=30000
SEED=1
PROJECT="varshare-timing-mt10-v3"
ANALYSIS_DIR="/netscratch/$USER/varshare/analysis_timing_v3"

# Define runs as: SCRIPT|ALGO_OR_VARIANT|HIDDEN_DIM|EXTRA_ARGS
RUNS=(
    # VarShare variants (train_routing.py)
    "routing|base|400|"
    "routing|routing|400|"
    # Standard baselines (train_baselines.py) — [400,400,400]
    "baselines|shared|400|"
    "baselines|independent|400|"
    "baselines|pcgrad|400|"
    "baselines|cagrad|400|"
    # Expert/Module baselines — full-size [400,400,400]
    "baselines|paco|400|--num-experts 5"
    "baselines|soft_mod|400|--num-modules 2"
    "baselines|moore|400|--num-experts 4"
    # Expert/Module baselines — param-matched [400/sqrt(K)]
    "baselines|paco|179|--num-experts 5"
    "baselines|soft_mod|283|--num-modules 2"
    "baselines|moore|200|--num-experts 4"
)

# Get this task's config
IDX=$((SLURM_ARRAY_TASK_ID - 1))
IFS='|' read -r SCRIPT ALGO HDIM EXTRA <<< "${RUNS[$IDX]}"

echo "=============================================="
echo "TIMING TEST V3: Task $SLURM_ARRAY_TASK_ID / 12"
echo "Script: train_${SCRIPT}.py"
echo "Algo/Variant: $ALGO"
echo "Hidden Dim: $HDIM"
echo "Extra Args: $EXTRA"
echo "=============================================="

EXP_NAME="timing_${ALGO}_h${HDIM}"

if [ "$SCRIPT" = "routing" ]; then
    python train_routing.py \
        --variant $ALGO \
        --hidden-dim $HDIM \
        --num-layers 3 \
        --total-timesteps $STEPS \
        --num-envs $NUM_ENVS \
        --eval-freq $EVAL_FREQ \
        --seed $SEED \
        --env-type metaworld \
        --mt-setting MT10 \
        --wandb-project $PROJECT \
        --exp-name $EXP_NAME \
        --analysis-dir $ANALYSIS_DIR \
        $EXTRA
else
    python train_baselines.py \
        --algo $ALGO \
        --hidden-dim $HDIM \
        --total-timesteps $STEPS \
        --num-envs $NUM_ENVS \
        --eval-freq $EVAL_FREQ \
        --seed $SEED \
        --env-type metaworld \
        --mt-setting MT10 \
        --wandb-project $PROJECT \
        --exp-name $EXP_NAME \
        --analysis-dir $ANALYSIS_DIR \
        $EXTRA
fi

echo ""
echo "TIMING TEST V3 COMPLETE: $EXP_NAME"
echo "Wall-clock: $SECONDS seconds"
