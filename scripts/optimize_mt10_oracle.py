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
    # 1. Sample Hyperparams
    params = get_trial_params(trial, "oracle")
    
    # Oracle trains on a SINGLE task (e.g. Task 0: Reach)
    # We should probably tune on a "Medium" task, not easiest.
    # Meta-World MT10 Task 0 is usually Reach (easiest).
    # Task 1 might be Push or Pick-Place.
    # Let's optimize on Task 0 for speed/stability, assuming transfer of params.
    TASK_ID = "0" 
    
    config = [
        "--algo", "oracle",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--task-id", TASK_ID,
        "--total-timesteps", "1000000",
        "--num-envs", "4",
        "--seed", "1",
        "--exp-name", f"optuna/oracle/trial_{trial.number}",
        "--eval-freq", "50000"
    ]
    
    config.extend(["--lr-actor", str(params["learning_rate_actor"])])
    config.extend(["--lr-critic", str(params["learning_rate_critic"])])
    config.extend(["--ent-coef", str(params["ent_coef"])])
    
    # Pass analysis dir
    if args and args.analysis_dir:
        config.extend(["--analysis-dir", args.analysis_dir])
        base_dir = args.analysis_dir
    else:
        base_dir = "analysis"

    sys.argv = ["train_baseline_ppo.py"] + config
    train_args = parse_args()
    
    try:
        history = train_func()
        
        save_dir = os.path.join(base_dir, f"optuna/oracle/trial_{trial.number}")
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
        study_name="mt10_oracle",
        storage=get_hpo_storage(base_dir=args.analysis_dir),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=args.n_trials)
