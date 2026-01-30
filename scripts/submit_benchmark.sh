#!/bin/bash
# Submit Hardware Benchmark Job
# 1. Requests a GPU Node
# 2. Runs Training on GPU
# 3. Runs Training on CPU (Same Node)

#SBATCH --job-name=bench_gpu_cpu
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Setup
source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
RESULTS_DIR="/netscratch/$USER/varshare/analysis/benchmark_hardware"
mkdir -p "$RESULTS_DIR"
mkdir -p logs

echo ">>> Starting Hardware Benchmark <<<"
echo "Node: $(hostname)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d ':' -f 2 | xargs)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "---------------------------------------------------"

TIMING_FILE="$RESULTS_DIR/times.txt"
rm -f "$TIMING_FILE"

# Function to run and time
run_test() {
    MODE=$1
    CUDA_FLAG=$2
    
    echo ">>> Running $MODE Training (20k steps) <<<"
    
    start_time=$(date +%s)
    
    python scripts/train_varshare_ppo.py \
        --env-type metaworld \
        --mt-setting MT10 \
        --total-timesteps 20000 \
        --cuda $CUDA_FLAG \
        --hidden-dim 256 \
        --num-envs 8 \
        --exp-name "benchmark_$MODE" \
        --analysis-dir "$RESULTS_DIR" \
        --seed 777 > "$RESULTS_DIR/${MODE}_log.txt" 2>&1
        
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "$MODE: $duration seconds" >> "$TIMING_FILE"
    echo "$MODE finished in $duration seconds."
}

# 1. Run GPU Test
run_test "GPU" "true"

# 2. Run CPU Test
run_test "CPU" "false"

echo "---------------------------------------------------"
echo ">>> Results Summary <<<"
cat "$TIMING_FILE"

# Calculate Speedup
gpu_time=$(grep "GPU" "$TIMING_FILE" | awk '{print $2}')
cpu_time=$(grep "CPU" "$TIMING_FILE" | awk '{print $2}')

if [ "$gpu_time" -gt 0 ]; then
    speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
    echo "Speedup (CPU Time / GPU Time): ${speedup}x"
else
    echo "Error: Invalid GPU time."
fi
echo "---------------------------------------------------"
echo "Results saved to: $TIMING_FILE"
