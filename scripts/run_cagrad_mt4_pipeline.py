import os
import subprocess
import optuna
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.hpo_utils import run_trial, get_trial_params

MT4_HPO_TIMESTEPS = 1000000 
MT4_FULL_TIMESTEPS = 10000000
TOTAL_SEEDS = [1, 2, 3]

def optimize_baseline_cagrad(trial):
    params = get_trial_params(trial, "cagrad")
    
    cmd = [
        "python", "scripts/train_baseline_ppo.py",
        "--algo", "cagrad",
        "--env-type", "metaworld",
        "--mt-setting", "MT4",
        "--exp-name", f"hpo_cagrad_mt4_trial_{trial.number}",
        "--total-timesteps", str(MT4_HPO_TIMESTEPS),
        "--lr-actor", str(params["learning_rate_actor"]),
        "--lr-critic", str(params["learning_rate_critic"]),
        "--ent-coef", str(params["ent_coef"]),
        "--cagrad-c", str(params["cagrad_c"]),
        "--batch-size", "128", 
        "--n-steps", "1024",
        "--eval-freq", "25000"
    ]
    
    return run_trial(trial, cmd)

def optimize_det_cagrad(trial):
    params = get_trial_params(trial, "det_cagrad")
    
    variant = "cagrad"
    kl_beta = str(0.05)
    rho_init = str(-4.0)

    cmd = [
        "python", "deterministic/scripts/train_det_ppo.py",
        "--variant", variant,
        "--env-type", "metaworld",
        "--mt-setting", "MT4",
        "--exp-name", f"hpo_det_cagrad_mt4_trial_{trial.number}",
        "--total-timesteps", str(MT4_HPO_TIMESTEPS),
        "--lr-actor", str(params["learning_rate_actor"]),
        "--lr-critic", str(params["learning_rate_critic"]),
        "--ent-coef", str(params["ent_coef"]),
        "--cagrad-c", str(params["cagrad_c"]),
        "--kl-beta", kl_beta,
        "--rho-init", rho_init,
        "--batch-size", "128", 
        "--n-steps", "1024",
        "--eval-freq", "25000"
    ]
    
    return run_trial(trial, cmd)

def submit_slurm_job(job_name, cmd_list, output_file):
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=/netscratch/{os.environ.get('USER', 'user')}/varshare/logs/{job_name}_%j.out
#SBATCH --error=/netscratch/{os.environ.get('USER', 'user')}/varshare/logs/{job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

source /netscratch/{os.environ.get('USER', 'user')}/varshare/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/varshare

{" ".join(cmd_list)}
"""
    script_path = f"/tmp/{job_name}.sh"
    with open(script_path, "w") as f:
        f.write(sbatch_content)
        
    print(f"Submitting SLURM job: {job_name}")
    subprocess.run(["sbatch", script_path])
    time.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hpo_base", "hpo_det", "full_evals"], required=True)
    args = parser.parse_args()
    
    os.makedirs(f"/netscratch/{os.environ.get('USER', 'user')}/varshare/logs", exist_ok=True)
    
    if args.mode == "hpo_base":
        study_base = optuna.create_study(study_name="cagrad_baseline_mt4", direction="maximize", storage="sqlite:///hpo.db", load_if_exists=True)
        study_base.optimize(optimize_baseline_cagrad, n_trials=1)
        
    elif args.mode == "hpo_det":
        study_det = optuna.create_study(study_name="det_cagrad_mt4", direction="maximize", storage="sqlite:///hpo.db", load_if_exists=True)
        study_det.optimize(optimize_det_cagrad, n_trials=1)
        
    elif args.mode == "full_evals":
        print("Evaluating HPO Results...")
        # Make sure to load the DB
        study_base = optuna.load_study(study_name="cagrad_baseline_mt4", storage="sqlite:///hpo.db")
        study_det = optuna.load_study(study_name="det_cagrad_mt4", storage="sqlite:///hpo.db")
        
        best_base = study_base.best_params
        best_det = study_det.best_params
        print(f"Best Base CAGrad: {best_base}")
        print(f"Best Det CAGrad: {best_det}")

        analysis_path = f"/netscratch/{os.environ.get('USER', 'user')}/varshare/analysis/phase2_evals"
        os.makedirs(analysis_path, exist_ok=True)
        
        for seed in TOTAL_SEEDS:
            cmd_base = [
                "python", "scripts/train_baseline_ppo.py",
                "--algo", "cagrad",
                "--env-type", "metaworld",
                "--mt-setting", "MT4",
                "--seed", str(seed),
                "--exp-name", "mt4_full_cagrad",
                "--analysis-dir", analysis_path,
                "--total-timesteps", str(MT4_FULL_TIMESTEPS),
                "--lr-actor", str(best_base["lr_actor"]),
                "--lr-critic", str(best_base["lr_critic"]),
                "--ent-coef", str(best_base["ent_coef"]),
                "--cagrad-c", str(best_base["cagrad_c"]),
                "--batch-size", "128", 
                "--n-steps", "1024",
                "--eval-freq", "25000"
            ]
            submit_slurm_job(f"eval_base_cag_s{seed}", cmd_base, "dummy")
            
            cmd_det = [
                "python", "deterministic/scripts/train_det_ppo.py",
                "--variant", "cagrad",
                "--env-type", "metaworld",
                "--mt-setting", "MT4",
                "--seed", str(seed),
                "--exp-name", "mt4_full_det_cagrad",
                "--analysis-dir", analysis_path,
                "--total-timesteps", str(MT4_FULL_TIMESTEPS),
                "--lr-actor", str(best_det["lr_actor"]),
                "--lr-critic", str(best_det["lr_critic"]),
                "--ent-coef", str(best_det["ent_coef"]),
                "--cagrad-c", str(best_det["cagrad_c"]),
                "--kl-beta", "0.05",
                "--rho-init", "-4.0",
                "--batch-size", "128", 
                "--n-steps", "1024",
                "--eval-freq", "25000"
            ]
            submit_slurm_job(f"eval_det_cag_s{seed}", cmd_det, "dummy")
            
        print("Final Full Evals distributed seamlessly!")

if __name__ == "__main__":
    main()
