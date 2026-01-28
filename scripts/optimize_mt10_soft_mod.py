import optuna
import sys
import os
import argparse
import json
from hpo_utils import get_hpo_storage, get_trial_params, calculate_objective
from train_baseline_ppo import train as train_func
from train_baseline_ppo import parse_args

def objective(trial):
    import builtins
    args = getattr(builtins, "HPO_ARGS", None)
    params = get_trial_params(trial, "soft_mod")
    
    config = [
        "--algo", "soft_mod",
        "--num-modules", "4",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "1000000",
        "--num-envs", "8",
        "--seed", "1",
        "--exp-name", f"optuna/soft_mod/trial_{trial.number}",
        "--eval-freq", "50000"
    ]
    
    config.extend(["--lr-actor", str(params["learning_rate_actor"])])
    config.extend(["--lr-critic", str(params["learning_rate_critic"])])
    config.extend(["--ent-coef", str(params["ent_coef"])])
    # SoftMod Specific
    config.extend(["--lr-routing", str(params["lr_routing"])])
    
    # Pass analysis dir
    if args and args.analysis_dir:
        config.extend(["--analysis-dir", args.analysis_dir])
        base_dir = args.analysis_dir
    else:
        base_dir = "analysis"

    sys.argv = ["train_baseline_ppo.py"] + config
    train_args = parse_args()
    
    try:
        history = train_func(report_callback=trial.report)
        
        save_dir = os.path.join(base_dir, f"optuna/soft_mod/trial_{trial.number}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
            
        return calculate_objective(history)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Base directory for results")
    args = parser.parse_args()
    
    import builtins
    builtins.HPO_ARGS = args

    study = optuna.create_study(
        study_name="mt10_soft_mod",
        storage=get_hpo_storage(base_dir=args.analysis_dir),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=args.n_trials)
