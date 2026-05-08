import os
import sys
import argparse
import optuna
import subprocess
import signal
import re
import math
import numpy as np
from collections import deque

# --- Environment Settings ---
ENV_SETTINGS = {
    "CP": {
        "env_type": "IdenticalCartPole",
        "mt_setting": "None",
        "total_timesteps": 1000000,
        "n_steps": 256,
        "num_envs": 10,
        "eval_freq": 20000,
        "hidden_dim": 400
    },
    "LL": {
        "env_type": "MultiTaskLunarLander",
        "mt_setting": "None",
        "total_timesteps": 1000000,
        "n_steps": 256,
        "num_envs": 10,
        "eval_freq": 20000,
        "hidden_dim": 400
    },
    "MT10": {
        "env_type": "metaworld",
        "mt_setting": "MT10",
        "total_timesteps": 1000000,
        "n_steps": 256,
        "num_envs": 10,
        "eval_freq": 20000,
        "hidden_dim": 400
    }
}

METHODS = [
    "shared", "independent", "pcgrad", "cagrad",
    "paco", "soft_mod", "moore",
    "base", "routing"
]

def run_trial(trial, args):
    env_config = ENV_SETTINGS[args.env_key].copy()
    
    if args.test:
        print(">>> TEST MODE: Overriding parameters for complete execution verification")
        env_config["total_timesteps"] = 6000
        env_config["n_steps"] = 128
        env_config["num_envs"] = 2
        env_config["eval_freq"] = 2000
    
    # --- Universal Base Hyperparameters ---
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-2, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-2, log=True)
    eps_clip = trial.suggest_float("eps_clip", 0.05, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 3.0)
    
    # We maintain standard MLP hidden layers at 3 (representing [hidden_dim, hidden_dim, hidden_dim])
    num_layers = 3
    hidden_dim = env_config["hidden_dim"]
    
    # Method-specific setup and CLI arguments
    extra_args = []
    script = "train_baselines.py" # default script
    
    if args.method == "shared":
        extra_args += ["--algo", "shared"]
    elif args.method == "independent":
        extra_args += ["--algo", "independent"]
    elif args.method == "pcgrad":
        extra_args += ["--algo", "pcgrad"]
    elif args.method == "cagrad":
        cagrad_c = trial.suggest_float("cagrad_c", 0.1, 0.5)
        extra_args += ["--algo", "cagrad", "--cagrad-c", str(cagrad_c)]
    elif args.method == "paco":
        num_experts = trial.suggest_categorical("num_experts", [2, 3, 4])
        # Mathematically scale network width to preserve constant parametric volume (~330k)
        hidden_dim = int(400 / math.sqrt(num_experts))
        lr_weights = trial.suggest_float("lr_weights", 1e-4, 1e-2, log=True)
        extra_args += ["--algo", "paco", "--num-experts", str(num_experts), "--lr-weights", str(lr_weights)]
    elif args.method == "soft_mod":
        num_modules = trial.suggest_categorical("num_modules", [2, 3, 4])
        hidden_dim = int(400 / math.sqrt(num_modules))
        lr_routing = trial.suggest_float("lr_routing", 1e-4, 1e-2, log=True)
        extra_args += ["--algo", "soft_mod", "--num-modules", str(num_modules), "--lr-routing", str(lr_routing)]
    elif args.method == "moore":
        num_experts = trial.suggest_categorical("num_experts", [2, 3, 4])
        hidden_dim = int(400 / math.sqrt(num_experts))
        lr_weights = trial.suggest_float("lr_weights", 1e-4, 1e-2, log=True)
        extra_args += ["--algo", "moore", "--num-experts", str(num_experts), "--lr-weights", str(lr_weights)]
    elif args.method in ["base", "routing"]:
        script = "train_routing.py"
        lr_task_scale = trial.suggest_float("lr_task_scale", 0.01, 1.5, log=True)
        mu_l2_coef = trial.suggest_float("mu_l2_coef", 1e-4, 1e-1, log=True)
        
        extra_args += [
            "--variant", args.method,
            "--lr-task-scale", str(lr_task_scale),
            "--mu-l2-coef", str(mu_l2_coef)
        ]
        
        if args.method == "routing":
            routing_alpha = trial.suggest_float("routing_alpha", 0.05, 1.5)
            routing_lambda = trial.suggest_float("routing_lambda", 1e-4, 1e-1, log=True)
            extra_args += [
                "--routing-alpha", str(routing_alpha),
                "--routing-lambda", str(routing_lambda)
            ]

    # Determine robust python interpreter path (prioritize virtual environment on Windows)
    python_exe = sys.executable
    venv_py = os.path.join(".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        python_exe = venv_py
    elif os.path.exists(os.path.join("venv", "Scripts", "python.exe")):
        python_exe = os.path.join("venv", "Scripts", "python.exe")

    # Construct the final subprocess command
    study_name = f"hpo_{args.env_key}_{args.method}"
    cmd = [
        python_exe, script,
        "--env-type", env_config["env_type"],
        "--mt-setting", env_config["mt_setting"],
        "--total-timesteps", str(env_config["total_timesteps"]),
        "--n-steps", str(env_config["n_steps"]),
        "--num-envs", str(env_config["num_envs"]),
        "--eval-freq", str(env_config["eval_freq"]),
        "--eval-episodes", "3",
        "--lr-actor", str(lr_actor),
        "--lr-critic", str(lr_critic),
        "--eps-clip", str(eps_clip),
        "--batch-size", str(batch_size),
        "--max-grad-norm", str(max_grad_norm),
        "--hidden-dim", str(hidden_dim),
        "--seed", str(args.seed),
        "--exp-name", f"{study_name}_trial{trial.number}",
        "--wandb-project", f"varshare-hpo-{args.env_key.lower()}"
    ] + extra_args

    if script == "train_routing.py":
        cmd += ["--num-layers", str(num_layers)]

    if args.analysis_dir:
        cmd += ["--analysis-dir", args.analysis_dir]

    print(f"\n--- Launching Trial {trial.number} ---")
    print(f"Command: {' '.join(cmd)}")
    
    # Capture and parse outputs in real-time
    eval_rewards = deque(maxlen=5)
    current_step = 0
    final_rolling_reward = -10000.0
    
    # Pattern matching for stdout diagnostics
    step_pattern = re.compile(r">>> Running Evaluation at Step (\d+)\.\.\.")
    reward_pattern = re.compile(r"Eval Reward:\s*([^\s|]+)")
    
    try:
        # Create full parent environment mapping for Windows sockets safety
        env_vars = os.environ.copy()
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            env=env_vars,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="") # Direct echo for real-time monitoring
            
            # Match step marker
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
                
            # Match evaluation results
            reward_match = reward_pattern.search(line)
            if reward_match and current_step > 0:
                try:
                    reward_val = float(reward_match.group(1))
                    eval_rewards.append(reward_val)
                    
                    # Compute rolling-5 mean of evaluation rewards
                    rolling_mean = np.mean(eval_rewards)
                    final_rolling_reward = rolling_mean
                    
                    print(f"[Optuna Parse] Step: {current_step} | New Reward: {reward_val:.2f} | Rolling-5 Mean: {rolling_mean:.2f}")
                    
                    # Report intermediate score to Optuna
                    trial.report(rolling_mean, current_step)
                    
                    # Trigger pruning check (ASHA)
                    if trial.should_prune():
                        print(f"\n>>> [ASHA Prune Triggered] Terminating Trial {trial.number} at step {current_step}")
                        process.terminate()
                        process.wait()
                        raise optuna.TrialPruned()
                        
                except Exception as parse_err:
                    print(f"[Optuna Parse Warning] Error processing evaluation metrics: {parse_err}")
                
                current_step = 0 # Reset step marker
                
        ret_code = process.wait()
        if ret_code != 0:
            print(f"[Warning] Subprocess exited with return code {ret_code}")
            
        # Return the final rolling evaluation reward computed
        return final_rolling_reward if len(eval_rewards) > 0 else -10000.0
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"[Error] Trial failed with exception: {e}")
        return -10000.0

def main():
    parser = argparse.ArgumentParser(description="VarShare HPO Optuna Sweep Controller")
    parser.add_argument("--env-key", type=str, required=True, choices=ENV_SETTINGS.keys(), help="Target environment (CP, LL, MT10)")
    parser.add_argument("--method", type=str, required=True, choices=METHODS, help="Algorithm to tune")
    parser.add_argument("--n-trials", type=int, default=60, help="Total trials to optimize")
    parser.add_argument("--storage", type=str, default="sqlite:///test_hpo.db", help="SQLAlchemy connection URI")
    parser.add_argument("--seed", type=int, default=1, help="Seed used inside all subprocesses")
    parser.add_argument("--analysis-dir", type=str, default=None, help="Root folder for output logs")
    parser.add_argument("--test", action="store_true", help="Toggle fast local dry-run verification mode")
    args = parser.parse_args()
    
    study_name = f"hpo_{args.env_key}_{args.method}"
    
    # Establish connection and database journaling optimization (WAL) if using SQLite
    db_timeout = 60
    if args.storage.startswith("sqlite"):
        # Append timeout parameter for parallel write locks safety
        if "?" in args.storage:
            args.storage += f"&timeout={db_timeout}"
        else:
            args.storage += f"?timeout={db_timeout}"
            
        # Optimize SQLite journal mode to WAL beforehand
        try:
            db_file = args.storage.split(":///")[1].split("?")[0]
            if db_file:
                subprocess.run(["sqlite3", db_file, "PRAGMA journal_mode=WAL;"], capture_output=True)
                print(f"Set SQLite database {db_file} to WAL (Write-Ahead Logging) mode.")
        except Exception as sqlite_err:
            print(f"SQLite WAL optimization skipped: {sqlite_err}")
            
    storage = optuna.storages.RDBStorage(args.storage)
    
    # Use Asynchronous Successive Halving (ASHA) with min_resource=400k and reduction_factor=2
    # Milestones are dynamically hit at: 400,000 steps (40%), 800,000 steps (80%)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=400000,
        reduction_factor=2
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler()
    )
    
    print("======================================================")
    print(f"STUDY NAME: {study_name}")
    print(f"STORAGE: {args.storage}")
    print(f"PRUNER: Successive Halving (ASHA) — Min: 400k, Reduction: 2")
    print(f"TRIALS BUDGET: {args.n_trials}")
    print(f"SUBPROCESS SEED: {args.seed}")
    print("======================================================")
    
    study.optimize(lambda trial: run_trial(trial, args), n_trials=args.n_trials)
    
    print("\n======================================================")
    print("HPO OPTIMIZATION CAMPAIGN COMPLETE!")
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Rolling-5 Mean Evaluation Reward: {study.best_trial.value:.2f}")
    print("Best Parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  - {k}: {v}")
    print("======================================================")

if __name__ == "__main__":
    main()
