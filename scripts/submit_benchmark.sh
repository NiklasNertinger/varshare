#!/bin/bash
# Submit Hardware Benchmark Job
# 1. Requests a GPU Node
# 2. Runs Training on GPU
# 3. Runs Training on CPU (Same Node)
# 4. Saves logs to separate files for analysis

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

# 1. Run GPU Test
echo ">>> Phase 1: GPU Training (15k steps) <<<"
python scripts/train_varshare_ppo.py \
    --env-type metaworld \
    --mt-setting MT10 \
    --total-timesteps 15000 \
    --cuda true \
    --hidden-dim 256 \
    --num-envs 8 \
    --exp-name benchmark_gpu \
    --analysis-dir "$RESULTS_DIR" \
    --seed 777 > "$RESULTS_DIR/gpu_log.txt" 2>&1

echo "GPU Run Complete. (Log: $RESULTS_DIR/gpu_log.txt)"

# 2. Run CPU Test
echo ">>> Phase 2: CPU Training (15k steps) <<<"
python scripts/train_varshare_ppo.py \
    --env-type metaworld \
    --mt-setting MT10 \
    --total-timesteps 15000 \
    --cuda false \
    --hidden-dim 256 \
    --num-envs 8 \
    --exp-name benchmark_cpu \
    --analysis-dir "$RESULTS_DIR" \
    --seed 777 > "$RESULTS_DIR/cpu_log.txt" 2>&1

echo "CPU Run Complete. (Log: $RESULTS_DIR/cpu_log.txt)"
echo "---------------------------------------------------"

# 3. Quick Analysis
echo ">>> Results Summary <<<"
echo "Method | SPS (Approx)"
echo "-------|-------------"

python -c "
import re
import statistics

def get_sps(params):
    sps_vals = []
    with open(params, 'r') as f:
        for line in f:
            # Look for 'SPS: <number>' pattern, regardless of context
            match = re.search(r'SPS:\s*([\d\.]+)', line)
            if match:
                try:
                    val = float(match.group(1))
                    sps_vals.append(val)
                except: pass
    
    if not sps_vals: return 0.0
    # Ignore first 10% as warmup
    cutoff = max(1, int(len(sps_vals) * 0.1))
    if len(sps_vals) <= cutoff: return statistics.mean(sps_vals)
    return statistics.mean(sps_vals[cutoff:])

gpu = get_sps('$RESULTS_DIR/gpu_log.txt')
cpu = get_sps('$RESULTS_DIR/cpu_log.txt')
speedup = gpu / cpu if cpu > 0 else 0

print(f'GPU SPS: {gpu:.2f}')
print(f'CPU SPS: {cpu:.2f}')
print(f'Speedup: {speedup:.2f}x')
" > "$RESULTS_DIR/summary_report.txt"

cat "$RESULTS_DIR/summary_report.txt"
echo "---------------------------------------------------"
echo ">>> Debug: Last 10 lines of GPU Log <<<"
tail -n 10 "$RESULTS_DIR/gpu_log.txt"
echo "---------------------------------------------------"
echo "Results saved to: $RESULTS_DIR/summary_report.txt"
