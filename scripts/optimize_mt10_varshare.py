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
    # Fixed Settings for MT10 HPO
    config = [
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "1000000",
        "--num-envs", "4",
        "--seed", str(trial.number),
        "--seed", "1",
        "--exp-name", f"optuna/varshare/trial_{trial.number}",
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
    # We need to re-parse args inside the objective function context? 
    # The 'args' here is from the outer scope, which is bad practice but works if we parse inside.
    # No, wait. 'args' for train_func comes from parse_args(). 
    # parse_args() uses sys.argv. So modifying sys.argv is correct.
    
    train_args = parse_args()
    
    try:
        history = train_func()
        
        # Save params to the correct location
        save_dir = os.path.join(base_dir, f"optuna/varshare/trial_{trial.number}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
            
        score = calculate_objective(history)
        return score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Base directory for results")
    args = parser.parse_args()

    # Make args available to objective function (global/closure hack)
    # Better: use partial or a class, but for now we rely on the closure scope if defined inside?
    # Actually, 'args' is global in this module scope if we define it in main. 
    # But multiprocessing might break globals.
    # To be safe, we should use a partial or class. 
    # Let's use a class or global variable injection.
    
    # Simple Global Injection for now as these are scripts.
    import builtins
    builtins.HPO_ARGS = args

    study = optuna.create_study(
        study_name="mt10_varshare",
        storage=get_hpo_storage(base_dir=args.analysis_dir),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"Starting HPO for VarShare. Storage: {get_hpo_storage()}")
    print(f"Running {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)
