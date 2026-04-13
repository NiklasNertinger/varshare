import os
import subprocess
import optuna
import sys
import time

# Adjust importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.hpo_utils import run_trial, get_trial_params

MT4_HPO_TIMESTEPS = 1000000 
MT4_FULL_TIMESTEPS = 10000000
N_TRIALS = 15
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
    
    # DetVarShare optimal fixed params from previous studies
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
    """
    Submits a job to the SLURM cluster by generating a temporary sbatch file.
    """
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=/netscratch/{os.environ.get('USER', 'user')}/varshare/logs/{job_name}_%j.out
#SBATCH --error=/netscratch/{os.environ.get('USER', 'user')}/varshare/logs/{job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=RTXA6000,L40S,batch,RTX3090,A100-40GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

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
    print("====================================")
    print("   CAGRAD MT4 FULL PIPELINE V1      ")
    print("====================================")
    
    # PHASE 1: Baseline CAGrad HPO
    print("\n>>> Phase 1: Baseline CAGrad HPO phase starting ...")
    study_base = optuna.create_study(study_name="cagrad_baseline_mt4", direction="maximize", storage="sqlite:///hpo.db", load_if_exists=True)
    
    # We only run trials if we haven't already hit the threshold (allows resume)
    completed_trials = [t for t in study_base.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) < N_TRIALS:
        study_base.optimize(optimize_baseline_cagrad, n_trials=(N_TRIALS - len(completed_trials)))
    best_cagrad_params = study_base.best_params
    print(f"Best Baseline CAGrad Params: {best_cagrad_params}")
    
    
    # PHASE 2: DetVarShare CAGrad HPO
    print("\n>>> Phase 2: Deterministic VarShare CAGrad HPO phase starting ...")
    study_det = optuna.create_study(study_name="det_cagrad_mt4", direction="maximize", storage="sqlite:///hpo.db", load_if_exists=True)
    completed_trials_det = [t for t in study_det.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials_det) < N_TRIALS:
        study_det.optimize(optimize_det_cagrad, n_trials=(N_TRIALS - len(completed_trials_det)))
    best_det_cagrad_params = study_det.best_params
    print(f"Best Det CAGrad Params: {best_det_cagrad_params}")

    # PHASE 3: Launch Cluster Full Evaluator
    print("\n>>> Phase 3: Launching full 10M-step SLURM cluster jobs ...")
    os.makedirs("analysis/phase2_evals", exist_ok=True)
    
    for seed in TOTAL_SEEDS:
        # Launch Baseline CAGrad
        cmd_base = [
            "python", "scripts/train_baseline_ppo.py",
            "--algo", "cagrad",
            "--env-type", "metaworld",
            "--mt-setting", "MT4",
            "--seed", str(seed),
            "--exp-name", "mt4_full_cagrad",
            "--analysis-dir", "analysis/phase2_evals",
            "--total-timesteps", str(MT4_FULL_TIMESTEPS),
            "--lr-actor", str(best_cagrad_params["lr_actor"]),
            "--lr-critic", str(best_cagrad_params["lr_critic"]),
            "--ent-coef", str(best_cagrad_params["ent_coef"]),
            "--cagrad-c", str(best_cagrad_params["cagrad_c"]),
            "--batch-size", "128", 
            "--n-steps", "1024",
            "--eval-freq", "25000"
        ]
        submit_slurm_job(f"mt4_base_cagrad_{seed}", cmd_base, f"mt4_cagrad_seed{seed}.log")
        
        # Launch DetVarShare CAGrad
        cmd_det = [
            "python", "deterministic/scripts/train_det_ppo.py",
            "--variant", "cagrad",
            "--env-type", "metaworld",
            "--mt-setting", "MT4",
            "--seed", str(seed),
            "--exp-name", "mt4_full_det_cagrad",
            "--analysis-dir", "analysis/phase2_evals",
            "--total-timesteps", str(MT4_FULL_TIMESTEPS),
            "--lr-actor", str(best_det_cagrad_params["lr_actor"]),
            "--lr-critic", str(best_det_cagrad_params["lr_critic"]),
            "--ent-coef", str(best_det_cagrad_params["ent_coef"]),
            "--cagrad-c", str(best_det_cagrad_params["cagrad_c"]),
            "--kl-beta", "0.05",
            "--rho-init", "-4.0",
            "--batch-size", "128", 
            "--n-steps", "1024",
            "--eval-freq", "25000"
        ]
        submit_slurm_job(f"mt4_det_cagrad_{seed}", cmd_det, f"mt4_det_cagrad_seed{seed}.log")
        
    print("\nAll cluster jobs successfully submitted! SLURM is handling the 10M parameter training.")
    print("\n>>> Phase 4: Final Plotting.")
    print("Once jobs complete, plots will automatically incorporate the new CAGrad algorithms since they are dumped into phase2_evals.")
    print("To manually trigger plotting immediately after finishing, run:")
    print("python scripts/plot_final_campaign.py --analysis-dir analysis/phase2_evals")
    
if __name__ == "__main__":
    main()
