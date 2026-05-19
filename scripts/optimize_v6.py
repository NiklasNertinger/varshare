import optuna
import argparse
import subprocess
import os
import re
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="Method to optimize (e.g., shared, paco, soft_mod, cagrad, base, routing)")
    parser.add_argument("--trials", type=int, default=80, help="Number of trials per method")
    parser.add_argument("--env-type", type=str, default="metaworld", help="Environment type")
    parser.add_argument("--mt-setting", type=str, default="MT10", help="MT setting")
    return parser.parse_args()

def objective(trial, args):
    # 1. Shared Hyperparameters (optimized for ALL methods)
    lr_actor = trial.suggest_float("lr-actor", 1e-5, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr-critic", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent-coef", 0.001, 0.05, log=True)
    eps_clip = trial.suggest_float("eps-clip", 0.1, 0.5)

    # Base commands
    base_cmd = [
        "python", 
        "train_routing.py" if args.method in ["base", "routing", "det_pcgrad"] else "train_baselines.py",
        "--total-timesteps", "10000000",
        "--n-steps", "256",
        "--batch-size", "256",
        "--num-envs", "10",
        "--eval-mode", "True",
        "--eval-freq", "50000",
        "--eval-episodes", "3",
        "--env-type", args.env_type,
        "--mt-setting", args.mt_setting,
        "--hidden-dim", "256",
        "--wandb-project", "varshare-hpo-mt10",
        "--exp-name", f"hpo_{args.method}_trial_{trial.number}",
        "--lr-actor", str(lr_actor),
        "--lr-critic", str(lr_critic),
        "--ent-coef", str(ent_coef),
        "--eps-clip", str(eps_clip),
        "--max-grad-norm", "2.0" # Fixed as discussed
    ]

    # Assign algorithm/variant flags
    if args.method in ["base", "routing", "det_pcgrad"]:
        variant_name = "pcgrad" if args.method == "det_pcgrad" else args.method
        base_cmd.extend(["--variant", variant_name])
        base_cmd.extend(["--scale-dim", "True"]) # Divided by sqrt(2)
    else:
        base_cmd.extend(["--algo", args.method])

    # 2. Method-Specific Hyperparameters
    if args.method in ["base", "routing", "det_pcgrad"]:
        mu_l2_coef = trial.suggest_float("mu-l2-coef", 1e-4, 1e-1, log=True)
        lr_task_scale = trial.suggest_float("lr-task-scale", 0.01, 2.0, log=True)
        base_cmd.extend(["--mu-l2-coef", str(mu_l2_coef), "--lr-task-scale", str(lr_task_scale)])

    if args.method == "routing":
        routing_alpha = trial.suggest_float("routing-alpha", 0.01, 1.5, log=True)
        routing_lambda = trial.suggest_float("routing-lambda", 1e-4, 1e-1, log=True)
        base_cmd.extend(["--routing-alpha", str(routing_alpha), "--routing-lambda", str(routing_lambda)])

    if args.method == "paco":
        lr_weights = trial.suggest_float("lr-weights", 1e-4, 5e-3, log=True)
        base_cmd.extend(["--lr-weights", str(lr_weights)])

    if args.method == "soft_mod":
        lr_routing = trial.suggest_float("lr-routing", 1e-4, 5e-3, log=True)
        base_cmd.extend(["--lr-routing", str(lr_routing)])

    if args.method == "cagrad":
        cagrad_c = trial.suggest_float("cagrad-c", 0.1, 0.8)
        base_cmd.extend(["--cagrad-c", str(cagrad_c)])

    # 3. Architecture Scaling K
    if args.method in ["paco", "care", "moore", "soft_mod"]:
        k = trial.suggest_int("num-experts", 2, 6)
        base_cmd.extend(["--scale-dim", "True"])
        if args.method == "soft_mod":
            base_cmd.extend(["--num-modules", str(k)])
        else:
            base_cmd.extend(["--num-experts", str(k)])
            
    # Subprocess tracking
    print(f"\n--- Starting Trial {trial.number} for {args.method} ---")
    
    process = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Regex patterns matching actual training script output:
    #   ">>> Running Evaluation at Step 50000..."
    #   "Eval Reward: 1187.08 | Eval Success: 0.10"
    step_regex = re.compile(r"Running Evaluation at Step (\d+)")
    eval_regex = re.compile(r"Eval Reward:\s*([-\d.]+)\s*\|\s*Eval Success:\s*([-\d.]+)")
    
    # Rolling-20 window of eval success rates for stable pruning signal
    success_window = deque(maxlen=20)
    best_success = float('-inf')
    current_eval_step = 0
    
    try:
        for line in process.stdout:
            print(line, end="") # Stream to SLURM logs
            
            # Track which step we're evaluating at
            step_match = step_regex.search(line)
            if step_match:
                current_eval_step = int(step_match.group(1))
            
            # Capture the eval success rate
            eval_match = eval_regex.search(line)
            if eval_match and current_eval_step > 0:
                success = float(eval_match.group(2))
                success_window.append(success)
                rolling_success = sum(success_window) / len(success_window)
                
                if rolling_success > best_success:
                    best_success = rolling_success
                    
                trial.report(rolling_success, current_eval_step)
                if trial.should_prune():
                    print(f"\n[Optuna] Trial {trial.number} PRUNED at step {current_eval_step} with rolling-20 success {rolling_success:.4f}.")
                    process.terminate()
                    process.wait()
                    raise optuna.exceptions.TrialPruned()
                    
        process.wait()
        if process.returncode != 0:
            print(f"\n[WARNING] Subprocess crashed with exit code {process.returncode}")
            raise optuna.exceptions.TrialPruned()
            
    except Exception as e:
        if not isinstance(e, optuna.exceptions.TrialPruned):
            print(f"\n[ERROR] Trial execution failed: {e}")
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise e

    return best_success

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs("context/HPOs", exist_ok=True)
    
    # Use JournalFileStorage instead of SQLite — designed for parallel cluster workers
    journal_path = f"context/HPOs/hpo_{args.method}.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_path)
    )
    
    # Pruner: ASHA with 2M step grace period
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=2000000, 
        reduction_factor=2
    )
    
    # Sampler: Multivariate TPE with constant liar for parallel cluster workers
    sampler = optuna.samplers.TPESampler(
        multivariate=True, 
        constant_liar=True
    )
    
    study = optuna.create_study(
        study_name=f"hpo_{args.method}_{args.mt_setting}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
        sampler=sampler
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    
    print(f"\n--- Study Completed for {args.method} ---")
    print("Best Trial:", study.best_trial.number)
    print("Best Rolling-20 Success:", study.best_value)
    print("Best Params:", study.best_params)
