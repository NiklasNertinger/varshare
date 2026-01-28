import optuna
import sys
import os
import argparse
import subprocess
import json
from hpo_utils import get_hpo_storage, get_trial_params, calculate_objective
from train_varshare_ppo import train as train_func
from train_varshare_ppo import parse_args

# Wrapper to call the training script programmatically
def objective(trial):
    import builtins
    args = getattr(builtins, "HPO_ARGS", None)
    
    # 1. Sample Hyperparams
    params = get_trial_params(trial, "varshare")
    
    # 2. Setup Arguments
    # We construct the equivalent of command line args
    # Fixed Settings for MT10 HPO (Scaled)
    config = [
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "2000000",   # INCREASED to 2M
        "--num-envs", "8",
        "--hidden-dim", "256",            # INCREASED to 256
        "--seed", "1",                    # Fixed seed for HPO stability
        "--exp-name", f"optuna_scaled/varshare/trial_{trial.number}",
        "--eval-freq", "50000"
    ]
    
    # HParams with specific flags
    config.extend(["--lr-actor", str(params["learning_rate_actor"])])
    config.extend(["--lr-critic", str(params["learning_rate_critic"])])
    config.extend(["--ent-coef", str(params["ent_coef"])])
    config.extend(["--kl-beta", str(params["kl_beta"])])
    config.extend(["--rho-init", str(params["rho_init"])])
    
    # Pass analysis dir if provided
    if "analysis_dir" in args and args.analysis_dir:
        config.extend(["--analysis-dir", args.analysis_dir])
        base_dir = args.analysis_dir
    else:
        base_dir = "analysis"

    sys.argv = ["train_varshare_ppo.py"] + config
    
    train_args = parse_args()
    
    try:
        history = train_func(report_callback=trial.report)
        
        # Save params to the correct location
        save_dir = os.path.join(base_dir, f"optuna_scaled/varshare/trial_{trial.number}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
            
        score = calculate_objective(history)
        return score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Prune failed trials
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Base directory for results")
    args = parser.parse_args()

    import builtins
    builtins.HPO_ARGS = args

    # Ensure storage dir exists
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    # New Storage File for Scaled Study
    storage_url = f"sqlite:///{os.path.join(args.analysis_dir, 'optuna_scaled.db')}"
    # Creating a Journal file is better for concurrency if on shared file system, 
    # but sqlite is standard. The original script used hpo_utils which defaults to journal.
    # Let's stick to Journal for consistency with hpo_utils but NEW file.
    storage_path = os.path.join(args.analysis_dir, "optuna_journal_scaled.log")
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(storage_path)
    )

    study = optuna.create_study(
        study_name="mt10_varshare_scaled", # NEW study name
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler()
    )
    
    print(f"Starting SCALED HPO for VarShare. Storage: {storage_path}")
    print(f"Running {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)
