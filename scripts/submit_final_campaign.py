import os
import json
import subprocess
import argparse

# --- Configuration ---
ENV_SETTINGS = {
    "CP": {"env_type": "IdenticalCartPole", "steps": 2000000, "eval_freq": 4000, "n_steps": 256, "hidden_dim": 64},
    "LL": {"env_type": "MultiTaskLunarLander", "steps": 10000000, "eval_freq": 20000, "n_steps": 256, "hidden_dim": 64},
    "MT4": {"env_type": "metaworld", "mt_setting": "MT4", "steps": 25000000, "eval_freq": 50000, "n_steps": 512, "hidden_dim": 256},
    "MT10": {"env_type": "metaworld", "mt_setting": "MT10", "steps": 5000000, "eval_freq": 50000, "n_steps": 512, "hidden_dim": 256}
}

# Slurm Config
PARTITION = "batch"
CPUS = 8
MEM = "16G"
SEEDS = [1, 2]

# Using num_envs 4 for all
NUM_ENVS = 4
BATCH_SIZE = 64

# --- Variants Definition ---
# Tuple: (Phase, Name, Base Script, Base Args, HPO Key to Look Up, HPO Db Version)
VARIANTS = [
    # --- PHASE 1 ---
    (1, "soft_mod", "train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"], "soft_mod", "v5"),
    (1, "shared_embedding", "train_baselines.py", ["--algo", "shared"], "shared_embedding", "v5"),
    (1, "paco", "train_baselines.py", ["--algo", "paco", "--num-experts", "5"], "paco", "v5"),
    (1, "care", "train_baselines.py", ["--algo", "care", "--num-experts", "6"], "care", "v5"),
    (1, "moore", "train_baselines.py", ["--algo", "moore", "--num-experts", "4"], "moore", "v5"),
    (1, "oracle", "train_baselines.py", ["--algo", "oracle"], "oracle", "v5"),
    (1, "shared_embedding_pcgrad", "train_baselines.py", ["--algo", "pcgrad"], "shared_embedding_pcgrad", "v5"),
    (1, "shared_embedding_cagrad", "train_baselines.py", ["--algo", "cagrad"], "cagrad", "v5"),
    (1, "det_base", "train_routing.py", ["--variant", "base"], "det_base", "v5"),
    (1, "det_routing", "train_routing.py", ["--variant", "routing"], "det_routing", "v5"),
    
    # --- PHASE 2 ---
    
    # Legacy specific configurations - HPO keys ignored here, we force them manually later
    
    (2, "det_lora", "train_routing.py", ["--variant", "lora"], "det_lora", "v5"),
    (2, "det_ara", "train_routing.py", ["--variant", "ara"], "det_ara", "v5"),
    (2, "det_l1", "train_routing.py", ["--variant", "l1"], "det_l1", "v5"),
    (2, "det_gated", "train_routing.py", ["--variant", "gated"], "det_gated", "v5"),
    (2, "det_pcgrad", "train_routing.py", ["--variant", "pcgrad"], "det_pcgrad", "v5"),
    (2, "det_cagrad", "train_routing.py", ["--variant", "cagrad"], "det_cagrad", "v5")
]

def load_hparams():
    with open("context/HPOs/best_hparams.json", "r") as f:
        return json.load(f)

def construct_hparam_args(hpo_dict):
    args = []
    for k, v in hpo_dict.items():
        # Convert underscores back to hyphens
        arg_name = f"--{k.replace('_', '-')}"
        args.extend([arg_name, str(v)])
    return args

def submit_job(phase, env_key, name, script, base_args, hpo_args, hidden_dim, steps, n_steps, eval_freq):
    # Construct base command string
    full_cmd = f"python {script} "
    
    # Base configuration
    config_args = [
        "--exp-name", f"final_phase{phase}_{name}",
        "--total-timesteps", str(steps),
        "--n-steps", str(n_steps),
        "--batch-size", str(BATCH_SIZE),
        "--num-envs", str(NUM_ENVS),
        "--eval-mode", "True",
        "--eval-freq", str(eval_freq),
        "--env-type", ENV_SETTINGS[env_key]["env_type"],
        "--hidden-dim", str(hidden_dim),
        "--analysis-dir", f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/analysis/final_eval/phase{phase}/{env_key}/{name}"
    ]
    
    if "mt_setting" in ENV_SETTINGS[env_key]:
        config_args.extend(["--mt-setting", ENV_SETTINGS[env_key]["mt_setting"]])
    
    # Combine everything
    all_args = base_args + config_args + hpo_args
    full_cmd += " ".join(all_args)
    
    # Setup Slurm output structure
    user = os.environ.get('USER', 'nertinger')
    log_dir = f"/netscratch/{user}/varshare/logs/final_eval/phase{phase}/{env_key}/{name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Slurm Time Dynamic
    slurm_time = "72:00:00" if env_key in ["MT10", "MT4"] else "24:00:00"
    
    # Oracle array math handling
    if name == "oracle":
        num_tasks = 10 if env_key == "MT10" else (4 if env_key == "MT4" else 1)
        total_jobs = len(SEEDS) * num_tasks
        array_str = f"0-{total_jobs - 1}"
        # We also reduce steps for oracle since it's single task, e.g. 5M steps instead of 50M
        # But we keep it as passed in, or maybe scale it? The script handles total_timesteps.
        # We will override steps for oracle to be steps // num_tasks
        steps = steps // num_tasks
        full_cmd = full_cmd.replace(f"--total-timesteps {steps * num_tasks}", f"--total-timesteps {steps}")
        
        seed_logic = f"""
SEED=$(($SLURM_ARRAY_TASK_ID / {num_tasks} + 1))
TASK_ID=$(($SLURM_ARRAY_TASK_ID % {num_tasks}))
echo "Starting Seed $SEED | Task $TASK_ID"
{full_cmd} --seed $SEED --task-id $TASK_ID
EXIT_CODE=$?
"""
    else:
        array_str = f"1-{len(SEEDS)}"
        seed_logic = f"""
echo "Starting Seed $SLURM_ARRAY_TASK_ID"
{full_cmd} --seed $SLURM_ARRAY_TASK_ID
EXIT_CODE=$?
"""
    
    # Write the sbatch script
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=f{phase}_{env_key}_{name}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.err
#SBATCH --partition={PARTITION}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={slurm_time}
#SBATCH --signal=B:USR1@120
#SBATCH --array={array_str}

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

{seed_logic}

if [ $EXIT_CODE -eq 99 ]; then
    echo "Job caught SIGUSR1 and checkpointed safely. Requeueing..."
    scontrol requeue $SLURM_JOB_ID
fi
"""

    script_path = f"logs/temp_submit_{phase}_{env_key}_{name}.sh"
    os.makedirs("logs", exist_ok=True)
    with open(script_path, "w") as f:
        f.write(sbatch_script)
        
    # Submit 
    print(f"Submitting: Phase {phase} | {env_key} | {name}")
    subprocess.run(["sbatch", script_path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=list(ENV_SETTINGS.keys()), help="Environments to submit")
    parser.add_argument("--algo-filter", type=str, default=None, help="Filter to only run a specific variant name")
    args = parser.parse_args()
    
    hparams_db = load_hparams()
    
    print("==================================================")
    print("   Submitting Final Evaluation Campaign to SLURM  ")
    print("==================================================")
    
    for target_phase in [1, 2]:
        print(f"\n{'='*50}")
        print(f"   SUBMITTING PHASE {target_phase} ALGORITHMS ")
        print(f"{'='*50}")
        
        for env_key in args.envs:
            print(f"\n>>> Preparing Environment: {env_key}")
            env_config = ENV_SETTINGS[env_key]
            
            for variant in VARIANTS:
                phase, name, script, base_args, hpo_key, hpo_version = variant
                
                # Enforce precise Phase 1 -> Phase 2 sequencing
                if phase != target_phase:
                    continue
                    
                if args.algo_filter and name != args.algo_filter:
                    continue
            
                # Special Handling for Legacy
                if hpo_key == "LEGACY":
                    # Only run legacy on MT4 & MT10
                    if env_key not in ["MT4", "MT10"]:
                        continue
                    # Hardcoded legacy hyperparams. Legacy uses its original un-constrained 2.0 max grad norm implicitly (we don't pass the new 1.0)
                    hpo_args = ["--kl-beta", "0.00043", "--lr-actor", "0.000117", "--lr-critic", "0.000259", "--rho-init", "-5.0", "--prior-scale", "1.0"]
                    hidden_dim_to_use = 64 # Special 64x64 backbone
                    grad_clip_arg = [] # Fallback to whatever defaults existed
                else:
                    hidden_dim_to_use = env_config["hidden_dim"]
                    grad_clip_arg = ["--max-grad-norm", "1.0"] # Conservative empirical clipping
                    
                    if name in ["oracle", "care", "moore"]:
                        # No HPO yet for these, provide robust defaults
                        hpo_args = ["--lr-actor", "3e-4", "--lr-critic", "3e-4"]
                        print(f"    Using default LR for {name} on {env_key}.")
                    else:
                        # Extract HPOs
                        try:
                            # MT10 natively uses MT4 params due to extraction parsing logic (we copied it over)
                            best_params = hparams_db[hpo_version][env_key].get(hpo_key, None)
                            
                            if not best_params:
                                print(f"    WARNING: No HPO found for {name} on {env_key}. Skipping!")
                                continue
                                
                            hpo_args = construct_hparam_args(best_params)
                        except KeyError:
                            print(f"    WARNING: HPO path missing for {name} on {env_key}. Skipping!")
                            continue
                
                base_args_with_clip = base_args + grad_clip_arg
                
                # Dispatch
                submit_job(
                    phase=phase,
                    env_key=env_key,
                    name=name,
                    script=script,
                    base_args=base_args_with_clip,
                    hpo_args=hpo_args,
                    hidden_dim=hidden_dim_to_use,
                    steps=env_config["steps"],
                    n_steps=env_config["n_steps"],
                    eval_freq=env_config["eval_freq"]
                )

if __name__ == "__main__":
    main()
