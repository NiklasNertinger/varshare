import os
import sys
import argparse
import optuna
import subprocess
import signal
import re

# --- Configuration ---
ENV_SETTINGS = {
    "CP": {
        "env_type": "IdenticalCartPole",
        "mt_setting": "None",
        "total_timesteps": 800000,
        "n_steps": 256,
        "num_envs": 1, # Using 1 for stable evaluation unless vectorized
        "eval_freq": 10000,
        "hidden_dim": 64
    },
    "LL": {
        "env_type": "MultiTaskLunarLander",
        "mt_setting": "None",
        "total_timesteps": 1500000,
        "n_steps": 256,
        "num_envs": 1,
        "eval_freq": 25000,
        "hidden_dim": 64
    },
    "MT4": {
        "env_type": "metaworld",
        "mt_setting": "MT4",
        "total_timesteps": 2500000,
        "n_steps": 512,
        "num_envs": 1,
        "eval_freq": 50000,
        "hidden_dim": 256
    },
    "MT10": {
        "env_type": "metaworld",
        "mt_setting": "MT10",
        "total_timesteps": 5000000,
        "n_steps": 512,
        "num_envs": 1,
        "eval_freq": 100000,
        "hidden_dim": 256
    }
}

DET_VARIANTS = [
    "det_base", "det_lora", "det_gated", "det_l1", "det_pcgrad", 
    "det_decay", "det_film", "det_hyperprior", "det_ara", "det_cagrad",
    "det_routing"
]
BASELINES = [
    "shared_embedding", "shared_embedding_pcgrad", 
    "paco", "soft_mod", "varshare_prior_opt", "cagrad"
]

METHODS = DET_VARIANTS + BASELINES

def run_trial(trial, args):
    env_config = ENV_SETTINGS[args.env_key].copy()
    
    if args.test:
        print(">>> TEST MODE: Reducing timesteps and steps for verification")
        env_config["total_timesteps"] = 2000
        env_config["n_steps"] = 256
        env_config["eval_freq"] = 1000
    
    # --- Universal Base Hyperparameters ---
    lr_actor = trial.suggest_float("lr_actor", 5e-5, 3e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-4, 5e-3, log=True)
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005, 0.01])
    
    cmd_base = [
        sys.executable, None, # Placeholder for script
        "--env-type", env_config["env_type"],
        "--mt-setting", env_config["mt_setting"],
        "--total-timesteps", str(env_config["total_timesteps"]),
        "--n-steps", str(env_config["n_steps"]),
        "--num-envs", str(env_config["num_envs"]),
        "--eval-freq", str(env_config["eval_freq"]),
        "--max-grad-norm", "2.0",
        "--lr-actor", str(lr_actor),
        "--lr-critic", str(lr_critic),
        "--ent-coef", str(ent_coef),
        "--seed", str(trial.number),
        "--hidden-dim", str(env_config["hidden_dim"])
    ]

    if args.analysis_dir:
        cmd_base += ["--analysis-dir", args.analysis_dir]

    study_name = f"v5_{args.env_key}_{args.method}"
    
    if args.method in BASELINES:
        if args.method == "varshare_prior_opt":
            script = "train_routing.py"
            cmd = cmd_base[:]
            cmd[1] = script
            rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
            kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
            prior_scale = trial.suggest_float("prior_scale", 1e-5, 1.0, log=True)
            cmd += [
                "--variant", "standard",
                "--rho-init", str(rho_init),
                "--kl-beta", str(kl_beta),
                "--prior-scale", str(prior_scale),
                "--embedding-type", "none",
                "--exp-name", f"{study_name}_trial{trial.number}"
            ]
        else:
            script = "train_baselines.py"
            cmd = cmd_base[:]
            cmd[1] = script
            cmd += ["--exp-name", f"{study_name}_trial{trial.number}"]
            
            if args.method == "soft_mod":
                lr_routing = trial.suggest_float("lr_routing", 1e-4, 5e-3, log=True)
                cmd += ["--algo", "soft_mod", "--num-modules", "4", "--lr-routing", str(lr_routing)]
            elif args.method == "paco":
                lr_weights = trial.suggest_float("lr_weights", 1e-4, 5e-3, log=True)
                cmd += ["--algo", "paco", "--num-experts", "4", "--lr-weights", str(lr_weights)]
            elif args.method == "shared_embedding":
                cmd += ["--algo", "shared"]
            elif args.method == "shared_embedding_pcgrad":
                cmd += ["--algo", "pcgrad"]
            elif args.method == "cagrad":
                cagrad_c = trial.suggest_categorical("cagrad_c", [0.2, 0.4, 0.6])
                cmd += ["--algo", "cagrad", "--cagrad-c", str(cagrad_c)]
                
    else:
        # Deterministic VarShare Variants
        script = "train_routing.py"
        cmd = cmd_base[:]
        cmd[1] = script
        cmd += ["--exp-name", f"{study_name}_trial{trial.number}"]
        
        # Extract variant name (e.g., det_base -> base)
        variant = args.method.split("_")[1]
        mu_l2_coef = trial.suggest_float("mu_l2_coef", 1e-5, 1e-1, log=True)
        mu_init = trial.suggest_float("mu_init", 1e-4, 1e-2, log=True)
        
        cmd += [
            "--variant", variant,
            "--mu-l2-coef", str(mu_l2_coef),
            "--mu-init", str(mu_init)
        ]
        
        if args.method == "det_cagrad":
            cagrad_c = trial.suggest_categorical("cagrad_c", [0.2, 0.4, 0.6])
            cmd += ["--cagrad-c", str(cagrad_c)]
        elif args.method == "det_routing":
            routing_alpha = trial.suggest_float("routing_alpha", 0.01, 1.0, log=True)
            routing_lambda = trial.suggest_float("routing_lambda", 1e-4, 1e-1, log=True)
            cmd += [
                "--routing-alpha", str(routing_alpha),
                "--routing-lambda", str(routing_lambda)
            ]

    print(f"Running Command: {' '.join(cmd)}")
    
    # Execution with real-time output parsing to report to Optuna
    final_reward = -1000.0
    current_step = 0
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Regex to catch: ">>> Running Evaluation at Step 256000..." and "Eval Reward: -158.30 | Eval Success: 0.00"
        step_pattern = re.compile(r">>> Running Evaluation at Step (\d+)\.\.\.")
        reward_pattern = re.compile(r"Eval Reward:\s*([^\s|]+)")
        final_reward_pattern = re.compile(r"FINAL_EVAL_REWARD:\s*([^\s]+)")
        
        for line in process.stdout:
            print(line, end="") # Echo to main terminal
            
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                
            reward_match = reward_pattern.search(line)
            if reward_match and current_step > 0:
                reward = float(reward_match.group(1))
                # Report intermediate value to Optuna, DO NOT Prune
                trial.report(reward, current_step)
                current_step = 0 # Reset until next evaluation
                
            final_match = final_reward_pattern.search(line)
            if final_match:
                final_reward = float(final_match.group(1))
                
        process.wait()
        
        if process.returncode != 0:
            print(f"Warning: Script failed with return code {process.returncode}")
            return final_reward
            
        return final_reward
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return final_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", type=str, required=True, choices=ENV_SETTINGS.keys())
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--storage", type=str, default="sqlite:///v5_hpo.db")
    parser.add_argument("--analysis-dir", type=str, default=None, help="Root analysis dir, e.g. /netscratch/$USER/varshare/analysis/det_hpo")
    parser.add_argument("--test", action="store_true", help="Run in fast verification mode")
    args = parser.parse_args()
    
    study_name = f"v5_{args.env_key}_{args.method}"
    
    print(f"Starting HPO Study: {study_name}")
    print(f"Environment: {args.env_key}")
    print(f"Method: {args.method}")
    print(f"Analysis Dir: {args.analysis_dir}")
    
    # Use hpo_utils for robust setup if available, else standard Optuna
    try:
        sys.path.append("scripts")
        from hpo_utils import get_hpo_storage
        storage = get_hpo_storage(args.storage)
    except ImportError:
        storage = args.storage
        
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler() # Removed seed=42 for parallel arrays
    )
    
    study.optimize(lambda trial: run_trial(trial, args), n_trials=args.n_trials)
    
    print("Optimization Complete.")
    print("Best Trial:")
    print(study.best_trial)

if __name__ == "__main__":
    main()
