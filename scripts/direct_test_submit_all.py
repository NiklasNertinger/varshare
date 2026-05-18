import os
import subprocess
import json

# Mapping of HPO JSON keys to the script and base arguments
KEY_TO_CMD = {
    "det_base": ("train_routing.py", ["--variant", "base"]),
    "det_lora": ("train_routing.py", ["--variant", "lora"]),
    "det_gated": ("train_routing.py", ["--variant", "gated"]),
    "det_l1": ("train_routing.py", ["--variant", "l1"]),
    "det_pcgrad": ("train_routing.py", ["--variant", "pcgrad"]),
    "det_decay": ("train_routing.py", ["--variant", "decay"]),
    "det_film": ("train_routing.py", ["--variant", "film"]),
    "det_hyperprior": ("train_routing.py", ["--variant", "hyperprior"]),
    "det_ara": ("train_routing.py", ["--variant", "ara"]),
    "routing": ("train_routing.py", ["--variant", "routing"]),
    
    # Baselines
    "shared_embedding": ("train_baselines.py", ["--algo", "shared"]),
    "shared_embedding_pcgrad": ("train_baselines.py", ["--algo", "pcgrad"]),
    "paco": ("train_baselines.py", ["--algo", "paco"]),
    "soft_mod": ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    "soft_modularization": ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    
    # Legacy Stochastic versions
    "varshare_standard": ("train.py", ["--variant", "standard"]),
    "varshare_scaled": ("train.py", ["--variant", "scaled"]),
    "varshare_bayes": ("train.py", ["--variant", "bayes"]),
    "varshare_partial": ("train.py", ["--variant", "partial"]),
    "varshare_prior_opt": ("train.py", ["--variant", "prior_opt"]),
}

def submit_all():
    hpo_file = "context/HPOs/best_hparams.json"
    
    if not os.path.exists(hpo_file):
        print(f"Error: {hpo_file} not found!")
        return

    with open(hpo_file, "r") as f:
        all_hparams = json.load(f)

    # The user asked for "HPO v1" or to bypass the DB error.
    # The static best_hparams.json file already contains the 'v5' tuning for ALL our algorithms (det_*, baselines)
    # So we will just load 'v5' from the static file directly, bypassing the sqlite database completely!
    hpo_version = "v5"
    env_key = "MT10"
    
    if hpo_version not in all_hparams or env_key not in all_hparams[hpo_version]:
        print(f"Error: Could not find {hpo_version} -> {env_key} in {hpo_file}")
        return

    hparams_db = all_hparams[hpo_version][env_key]

    NUM_ENVS = 10
    STEPS = 400000
    EVAL_FREQ = 20000
    BATCH_SIZE = 64
    
    print(f"Starting Submission for ALL Algorithms using HPO {hpo_version}...")
    
    submitted_count = 0
    for name, best_params in hparams_db.items():
        # Ignore "_consistent" variants as they are duplicates for the same method
        if name.endswith("_consistent"):
            continue
            
        if name not in KEY_TO_CMD:
            print(f"Skipping unknown method: {name}")
            continue
            
        script, base_args = KEY_TO_CMD[name]
        
        full_cmd = f"python {script} "
        
        config_args = [
            "--exp-name", f"direct_test_v4_{name}",
            "--total-timesteps", str(STEPS),
            "--n-steps", "512",
            "--batch-size", str(BATCH_SIZE),
            "--num-envs", str(NUM_ENVS),
            "--eval-mode", "True",
            "--eval-freq", str(EVAL_FREQ),
            "--env-type", "metaworld",
            "--mt-setting", "MT10",
            "--hidden-dim", "256",
            "--wandb-project", "varshare-v2-test",
            "--analysis-dir", f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/analysis/direct_test_v4/{name}"
        ]
        
        # Inject extracted HPOs
        for k, v in best_params.items():
            config_args.extend([f"--{k.replace('_', '-')}", str(v)])
                
        all_args = base_args + config_args
        full_cmd += " ".join(all_args)
        
        log_dir = f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/logs/direct_test_v4/{name}"
        os.makedirs(log_dir, exist_ok=True)
        
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=test_{name}
#SBATCH --output={log_dir}/%A.out
#SBATCH --error={log_dir}/%A.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

echo "Starting Direct Test for {name}"
{full_cmd} --seed 1
"""
        
        script_path = f"logs/temp_submit_v4_{name}.sh"
        os.makedirs("logs", exist_ok=True)
        with open(script_path, "w") as f:
            f.write(sbatch_script)
            
        print(f"Submitting Direct Test: {name}")
        subprocess.run(["sbatch", script_path])
        submitted_count += 1
        
    print("-" * 50)
    print(f"Successfully dispatched {submitted_count} jobs using {hpo_version} hyperparameters!")

if __name__ == "__main__":
    submit_all()
