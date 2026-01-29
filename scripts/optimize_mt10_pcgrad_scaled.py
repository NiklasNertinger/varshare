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
    
    # 1. Sample Hyperparams (PCGrad uses Shared/Standard params)
    params = get_trial_params(trial, "shared")
    
    config = [
        "--algo", "pcgrad", 
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", os.environ.get("HPO_TIME_STEPS", "2000000"),
        "--num-envs", "8",
        "--hidden-dim", "256",
        "--seed", "1",
        "--exp-name", f"optuna_scaled/pcgrad/trial_{trial.number}",
        "--eval-freq", "50000"
    ]
    
    config.extend(["--lr-actor", str(params["learning_rate_actor"])])
    config.extend(["--lr-critic", str(params["learning_rate_critic"])])
    config.extend(["--ent-coef", str(params["ent_coef"])])
    
    if "analysis_dir" in args and args.analysis_dir:
        config.extend(["--analysis-dir", args.analysis_dir])
        base_dir = args.analysis_dir
    else:
        base_dir = "analysis"

    sys.argv = ["train_baseline_ppo.py"] + config
    
    try:
        history = train_func(report_callback=trial.report)
        
        save_dir = os.path.join(base_dir, f"optuna_scaled/pcgrad/trial_{trial.number}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
            
        score = calculate_objective(history)
        return score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--analysis-dir", type=str, default="analysis")
    args = parser.parse_args()

    import builtins
    builtins.HPO_ARGS = args
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    storage_path = os.path.join(args.analysis_dir, "optuna_journal_scaled.log")
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))

    study = optuna.create_study(
        study_name="mt10_pcgrad_scaled",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=args.n_trials)
