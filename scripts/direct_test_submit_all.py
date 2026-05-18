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
    "det_routing": ("train_routing.py", ["--variant", "routing"]),
    
    # Baselines
    "shared_embedding": ("train_baselines.py", ["--algo", "shared"]),
    "shared_embedding_pcgrad": ("train_baselines.py", ["--algo", "pcgrad"]),
    "paco": ("train_baselines.py", ["--algo", "paco"]),
    "soft_mod": ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    "soft_modularization": ("train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"]),
}

DEFAULTS = {
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "eps_clip": 0.2,
    "max_grad_norm": 1.0,
    "batch_size": 64
}

def submit_all():
    v2_file = "context/HPOs/best_hparams_v2.json"
    fallback_file = "context/HPOs/best_hparams.json"
    
    hparams_v2 = {}
    if os.path.exists(v2_file):
        with open(v2_file, "r") as f:
            data = json.load(f)
            hparams_v2 = data.get("v5", {}).get("MT10", {})
            print(f"Loaded {len(hparams_v2)} fresh v2 hyperparameters.")
            
    hparams_fallback = {}
    if os.path.exists(fallback_file):
        with open(fallback_file, "r") as f:
            data = json.load(f)
            # Use v5 from fallback if available, or v4
            hparams_fallback = data.get("v5", {}).get("MT10", {})
            if not hparams_fallback:
                hparams_fallback = data.get("v4", {}).get("MT10", {})
            print(f"Loaded fallback hyperparameters for {len(hparams_fallback)} methods.")

    NUM_ENVS = 10
    STEPS = 400000
    EVAL_FREQ = 20000
    BATCH_SIZE = 512
    
    print("Starting Submission for ALL Algorithms...")
    
    # We want to run all algorithms listed in KEY_TO_CMD
    submitted_count = 0
    
    # To avoid duplicates if keys map to same command
    processed_methods = set()
    
    for name, (script, base_args) in KEY_TO_CMD.items():
        # Avoid processing duplicate aliases
        cmd_key = (script, tuple(base_args))
        if cmd_key in processed_methods:
            continue
        processed_methods.add(cmd_key)
        
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
        
        # Determine best params
        best_params = None
        if name in hparams_v2:
            print(f"  -> Using fresh v2 HPOs for {name}")
            best_params = hparams_v2[name]
        elif name in hparams_fallback:
            print(f"  -> Using fallback HPOs for {name}")
            best_params = hparams_fallback[name]
        else:
            print(f"  -> WARNING: No HPOs found for {name}. Using defaults.")
            best_params = DEFAULTS
            
        # Inject HPOs
        for k, v in best_params.items():
            config_args.extend([f"--{k.replace('_', '-')}", str(v)])
                
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
        submitted_count += 1
        print("-" * 50)
        
    print(f"Successfully dispatched {submitted_count} jobs!")

if __name__ == "__main__":
    submit_all()
