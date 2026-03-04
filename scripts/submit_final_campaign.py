import os
import json
import subprocess

# --- Configuration ---
ENV_SETTINGS = {
    "CP": {"env_type": "IdenticalCartPole", "steps": 2000000, "eval_freq": 20000, "n_steps": 256, "hidden_dim": 64},
    "LL": {"env_type": "MultiTaskLunarLander", "steps": 10000000, "eval_freq": 100000, "n_steps": 256, "hidden_dim": 64},
    "MT4": {"env_type": "metaworld", "mt_setting": "MT4", "steps": 25000000, "eval_freq": 250000, "n_steps": 512, "hidden_dim": 256},
    "MT10": {"env_type": "metaworld", "mt_setting": "MT10", "steps": 50000000, "eval_freq": 500000, "n_steps": 512, "hidden_dim": 256}
}

# Slurm Config
PARTITION = "batch"
CPUS = 8
MEM = "16G"
SEEDS = [1, 2, 3]

# Using num_envs 4 for all
NUM_ENVS = 4
BATCH_SIZE = 64

# --- Variants Definition ---
# Tuple: (Phase, Name, Base Script, Base Args, HPO Key to Look Up, HPO Db Version)
VARIANTS = [
    # --- PHASE 1 ---
    (1, "soft_mod", "scripts/train_baseline_ppo.py", ["--algo", "soft_mod", "--num-modules", "4"], "soft_modularization", "v5"),
    (1, "shared_embedding", "scripts/train_baseline_ppo.py", ["--algo", "shared"], "shared_embedding", "v5"),
    (1, "paco", "scripts/train_baseline_ppo.py", ["--algo", "paco", "--num-experts", "4"], "paco", "v5"),
    (1, "shared_embedding_pcgrad", "scripts/train_baseline_ppo.py", ["--algo", "pcgrad"], "shared_embedding_pcgrad", "v5"),
    (1, "varshare_prob_consistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "true"], "varshare_standard_consistent", "v4"),
    (1, "det_base", "deterministic/scripts/train_det_ppo.py", ["--variant", "base"], "det_base", "v5"),
    
    # --- PHASE 2 ---
    (2, "varshare_prob_inconsistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "false"], "varshare_standard", "v4"),
    (2, "varshare_reptile_consistent", "scripts/train_varshare_ppo.py", ["--variant", "reptile", "--consistent-noise", "true"], "varshare_standard_consistent", "v4"), # Approximate using standard
    (2, "varshare_partial_consistent", "scripts/train_varshare_ppo.py", ["--variant", "partial", "--consistent-noise", "true"], "varshare_partial_consistent", "v4"),
    (2, "varshare_lora_consistent", "scripts/train_varshare_ppo.py", ["--variant", "lora", "--consistent-noise", "true"], "varshare_standard_consistent", "v4"), # Approximate using standard
    
    # Legacy specific configurations - HPO keys ignored here, we force them manually later
    (2, "legacy_prob_base_consistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "true"], "LEGACY", None),
    (2, "legacy_prob_base_inconsistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "false"], "LEGACY", None),
    
    (2, "det_lora", "deterministic/scripts/train_det_ppo.py", ["--variant", "lora"], "det_lora", "v5"),
    (2, "det_ara", "deterministic/scripts/train_det_ppo.py", ["--variant", "ara"], "det_ara", "v5"),
    (2, "det_l1", "deterministic/scripts/train_det_ppo.py", ["--variant", "l1"], "det_l1", "v5"),
    (2, "det_gated", "deterministic/scripts/train_det_ppo.py", ["--variant", "gated"], "det_gated", "v5"),
    (2, "det_pcgrad", "deterministic/scripts/train_det_ppo.py", ["--variant", "pcgrad"], "det_pcgrad", "v5")
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
        "--analysis-dir", f"/netscratch/$USER/varshare/analysis/final_eval/phase{phase}/{env_key}/{name}"
    ]
    
    if "mt_setting" in ENV_SETTINGS[env_key]:
        config_args.extend(["--mt-setting", ENV_SETTINGS[env_key]["mt_setting"]])
    
    # Combine everything
    all_args = base_args + config_args + hpo_args
    full_cmd += " ".join(all_args)
    
    # Setup Slurm output structure
    log_dir = f"/netscratch/$USER/varshare/logs/final_eval/phase{phase}/{env_key}/{name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Slurm Time Dynamic
    slurm_time = "168:00:00" if env_key == "MT10" else ("72:00:00" if env_key == "MT4" else "24:00:00")
    
    # Write the sbatch script
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=f{phase}_{env_key}_{name}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.err
#SBATCH --partition={PARTITION}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={slurm_time}
#SBATCH --array=1-3 # 3 seeds

source /netscratch/$USER/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

echo "Starting Seed $SLURM_ARRAY_TASK_ID"

{full_cmd} --seed $SLURM_ARRAY_TASK_ID
"""

    script_path = f"logs/temp_submit_{phase}_{env_key}_{name}.sh"
    os.makedirs("logs", exist_ok=True)
    with open(script_path, "w") as f:
        f.write(sbatch_script)
        
    # Submit 
    print(f"Submitting: Phase {phase} | {env_key} | {name}")
    subprocess.run(["sbatch", script_path])


def main():
    hparams_db = load_hparams()
    
    print("==================================================")
    print("   Submitting Final Evaluation Campaign to SLURM  ")
    print("==================================================")
    
    for env_key in ENV_SETTINGS.keys():
        print(f"\n>>> Preparing Environment: {env_key}")
        env_config = ENV_SETTINGS[env_key]
        
        for variant in VARIANTS:
            phase, name, script, base_args, hpo_key, hpo_version = variant
            
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
