import os
import json
import subprocess
import argparse

ENV_SETTINGS = {
    "MT10": {"env_type": "metaworld", "mt_setting": "MT10", "steps": 400000, "eval_freq": 20000, "n_steps": 512, "hidden_dim": 256}
}

PARTITION = "batch"
CPUS = 8
MEM = "16G"
SEEDS = [1]

NUM_ENVS = 10
BATCH_SIZE = 512

VARIANTS = [
    (1, "soft_mod", "train_baselines.py", ["--algo", "soft_mod", "--num-modules", "4"], "soft_mod", "v5"),
    (1, "shared_embedding", "train_baselines.py", ["--algo", "shared"], "shared_embedding", "v5"),
    (1, "shared_embedding_pcgrad", "train_baselines.py", ["--algo", "pcgrad"], "shared_embedding_pcgrad", "v5"),
    (1, "det_routing", "train_routing.py", ["--variant", "routing"], "det_routing", "v5")
]

def load_hparams():
    with open("context/HPOs/best_hparams_v2.json", "r") as f:
        return json.load(f)

def construct_hparam_args(hpo_dict):
    args = []
    for k, v in hpo_dict.items():
        arg_name = f"--{k.replace('_', '-')}"
        args.extend([arg_name, str(v)])
    return args

def submit_job(phase, env_key, name, script, base_args, hpo_args, hidden_dim, steps, n_steps, eval_freq):
    full_cmd = f"python {script} "
    
    config_args = [
        "--exp-name", f"final_v2_phase{phase}_{name}",
        "--total-timesteps", str(steps),
        "--n-steps", str(n_steps),
        "--batch-size", str(BATCH_SIZE),
        "--num-envs", str(NUM_ENVS),
        "--eval-mode", "True",
        "--eval-freq", str(eval_freq),
        "--env-type", ENV_SETTINGS[env_key]["env_type"],
        "--hidden-dim", str(hidden_dim),
        "--wandb-project", "varshare-v2-test",
        "--analysis-dir", f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/analysis/final_eval_v2/phase{phase}/{env_key}/{name}"
    ]
    
    if "mt_setting" in ENV_SETTINGS[env_key]:
        config_args.extend(["--mt-setting", ENV_SETTINGS[env_key]["mt_setting"]])
    
    all_args = base_args + config_args + hpo_args
    full_cmd += " ".join(all_args)
    
    user = os.environ.get('USER', 'nertinger')
    log_dir = f"/netscratch/{user}/varshare/logs/final_eval_v2/phase{phase}/{env_key}/{name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Slurm Time Limit (24 hours for benchmarking runs)
    slurm_time = "24:00:00"
    
    array_str = f"1-{len(SEEDS)}"
    seed_logic = f"""
echo "Starting Seed $SLURM_ARRAY_TASK_ID"
{full_cmd} --seed $SLURM_ARRAY_TASK_ID
EXIT_CODE=$?
"""
    
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=f_v2_{env_key}_{name}
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

    script_path = f"logs/temp_submit_v2_{phase}_{env_key}_{name}.sh"
    os.makedirs("logs", exist_ok=True)
    with open(script_path, "w") as f:
        f.write(sbatch_script)
        
    print(f"Submitting [v2]: Phase {phase} | {env_key} | {name}")
    subprocess.run(["sbatch", script_path])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["MT10"], help="Environments to submit")
    args = parser.parse_args()
    
    hparams_db = load_hparams()
    
    print("==================================================")
    print("   Submitting Final Evaluation Campaign v2 to SLURM  ")
    print("==================================================")
    
    for env_key in args.envs:
        print(f"\n>>> Preparing Environment: {env_key}")
        env_config = ENV_SETTINGS[env_key]
        
        for variant in VARIANTS:
            phase, name, script, base_args, hpo_key, hpo_version = variant
            hidden_dim_to_use = env_config["hidden_dim"]
            grad_clip_arg = ["--max-grad-norm", "1.0"]
            
            try:
                best_params = hparams_db[hpo_version][env_key].get(hpo_key, None)
                if not best_params:
                    print(f"    WARNING: No HPO v2 found for {name} on {env_key}. Skipping!")
                    continue
                hpo_args = construct_hparam_args(best_params)
            except KeyError:
                print(f"    WARNING: HPO v2 path missing for {name} on {env_key}. Skipping!")
                continue
            
            base_args_with_clip = base_args + grad_clip_arg
            
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
