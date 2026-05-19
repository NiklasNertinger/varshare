"""
Standalone script to submit ONLY the det_pcgrad HPO to SLURM.
Use this when the other HPOs are already running.
"""
import os
import subprocess

METHOD = "det_pcgrad"
TRIALS = 80
MAX_CONCURRENT = 8
PARTITION = "RTX3090"
MEM = "16G"
CPUS = 8
TIME = "48:00:00"
ENV_TYPE = "metaworld"
MT_SETTING = "MT10"

log_dir = f"logs/hpo_v6/{METHOD}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("context/HPOs", exist_ok=True)

script_path = f"logs/submit_hpo_{METHOD}.sh"

sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=hpo_{METHOD}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.err
#SBATCH --partition={PARTITION}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={TIME}
#SBATCH --gpus=1
#SBATCH --array=1-{TRIALS}%{MAX_CONCURRENT}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare
export PYTHONUNBUFFERED=1

echo "Starting HPO Trial $SLURM_ARRAY_TASK_ID for {METHOD}"
python scripts/optimize_v6.py --method {METHOD} --trials 1 --env-type {ENV_TYPE} --mt-setting {MT_SETTING}
"""

with open(script_path, "w") as f:
    f.write(sbatch_content)

print(f"Submitting {TRIALS} trials (%{MAX_CONCURRENT} concurrent) for {METHOD} [{PARTITION}]")
subprocess.run(["sbatch", script_path])
