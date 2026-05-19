import os
import subprocess

# --- Config ---
TRIALS_PER_METHOD = 80
PARALLEL_WORKERS = 8 # Number of SLURM array jobs per method
TRIALS_PER_WORKER = TRIALS_PER_METHOD // PARALLEL_WORKERS

METHODS_CPU = ["shared", "paco", "soft_mod", "care", "moore", "pcgrad", "cagrad"]
METHODS_GPU = ["base", "routing"]

ENV_TYPE = "metaworld"
MT_SETTING = "MT10"

# --- SLURM Defaults ---
MEM = "16G"
CPUS = 8
TIME = "48:00:00" # Long time since they run sequential trials

def generate_and_submit_hpo(method, use_gpu):
    partition = "batch" # Use batch for everything as per submit_final_campaign_v2.py
    gpu_flag = "#SBATCH --gpus=1" if use_gpu else ""
    
    log_dir = f"logs/hpo_v6/{method}"
    os.makedirs(log_dir, exist_ok=True)
    
    script_path = f"logs/submit_hpo_{method}.sh"
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=hpo_{method}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.err
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={TIME}
{gpu_flag}
#SBATCH --array=1-{PARALLEL_WORKERS}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
export PYTHONUNBUFFERED=1

echo "Starting HPO Worker $SLURM_ARRAY_TASK_ID for {method}"
python scripts/optimize_v6.py --method {method} --trials {TRIALS_PER_WORKER} --env-type {ENV_TYPE} --mt-setting {MT_SETTING}
"""
    
    with open(script_path, "w") as f:
        f.write(sbatch_content)
        
    print(f"Submitting {PARALLEL_WORKERS} parallel HPO workers for {method} [{partition}]")
    subprocess.run(["sbatch", script_path])

if __name__ == "__main__":
    print("==================================================")
    print("   Submitting Final HPO Campaign to SLURM Cluster ")
    print("==================================================")
    
    os.makedirs("context/HPOs", exist_ok=True)
    
    for method in METHODS_CPU:
        generate_and_submit_hpo(method, use_gpu=False)
        
    for method in METHODS_GPU:
        generate_and_submit_hpo(method, use_gpu=True)
        
    print("\\nAll HPO jobs submitted successfully!")
