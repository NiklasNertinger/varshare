import os
import subprocess
import json

def extract_hparams():
    user = os.environ.get("USER", "nertinger")
    db_file = f"/netscratch/{user}/varshare/test_hpo_v2.db"
    
    print(f"Attempting to extract HPOs from {db_file}...")
    if not os.path.exists(db_file):
        print(f"CRITICAL ERROR: Database file {db_file} does not exist!")
        print("Cannot extract v2 HPOs. Falling back to default parameters.")
        return None
        
    try:
        import optuna
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_file}", engine_kwargs={"timeout": 60})
        studies = storage.get_all_studies()
    except Exception as e:
        print(f"CRITICAL ERROR reading Optuna database: {e}")
        return None

    method_mapping = {
        "shared": "shared_embedding",
        "pcgrad": "shared_embedding_pcgrad",
        "soft_mod": "soft_mod",
        "routing": "det_routing"
    }
    
    hparams_data = {}
    for study in studies:
        name = study.study_name
        if name.startswith("hpo_v2_MT10_"):
            method = name.replace("hpo_v2_MT10_", "")
            if method in method_mapping:
                target_key = method_mapping[method]
                try:
                    best_trial = storage.get_best_trial(study._study_id)
                    hparams_data[target_key] = best_trial.params
                    print(f"Successfully extracted champion for '{target_key}' with score: {best_trial.value:.2f}")
                except ValueError:
                    print(f"No completed trials for study: {name}.")
    
    return hparams_data

methods = [
    ("soft_mod", "train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    ("shared_embedding", "train_baselines.py", ["--algo", "shared"]),
    ("shared_embedding_pcgrad", "train_baselines.py", ["--algo", "pcgrad"]),
    ("det_routing", "train_routing.py", ["--variant", "routing"])
]

NUM_ENVS = 10
STEPS = 400000
EVAL_FREQ = 20000
BATCH_SIZE = 64

print("Starting Direct Test Submission...")
hparams_db = extract_hparams() or {}

for name, script, base_args in methods:
    full_cmd = f"python {script} "
    
    config_args = [
        "--exp-name", f"direct_test_{name}",
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
        "--analysis-dir", f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/analysis/direct_test/{name}"
    ]
    
    best_params = hparams_db.get(name, None)
    if best_params:
        print(f"Using extracted v2 HPOs for {name}!")
        for k, v in best_params.items():
            config_args.extend([f"--{k.replace('_', '-')}", str(v)])
    else:
        print(f"WARNING: No HPOs found for {name}. Using base parameters.")
            
    all_args = base_args + config_args
    full_cmd += " ".join(all_args)
    
    log_dir = f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/logs/direct_test/{name}"
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
    
    script_path = f"logs/temp_submit_direct_{name}.sh"
    os.makedirs("logs", exist_ok=True)
    with open(script_path, "w") as f:
        f.write(sbatch_script)
        
    print(f"Submitting Direct Test: {name}")
    subprocess.run(["sbatch", script_path])
    print("-" * 50)
