
import os
import sys
import time
import argparse
import optuna
import subprocess
import signal

# --- Configuration ---
ENV_SETTINGS = {
    "MT4": {
        "env_type": "metaworld",
        "mt_setting": "MT4",
        "total_timesteps": 2500000,
        "n_steps": 512,
        "num_envs": 4,
        "n_updates": 1220, # Approx 
        "eval_freq": 25000,
        "hidden_dim_standard": 256,
        "hidden_dim_scaled": 181
    },
    "LL": {
        "env_type": "MultiTaskLunarLander",
        "mt_setting": "None", # Ignored
        "total_timesteps": 1500000,
        "n_steps": 256,
        "num_envs": 4,
        "n_updates": 1464,
        "eval_freq": 15000,
        "hidden_dim_standard": 64,
        "hidden_dim_scaled": 45
    },
    "IdenticalLL": {
        "env_type": "IdenticalLunarLander",
        "mt_setting": "None",
        "total_timesteps": 1500000,
        "n_steps": 256,
        "num_envs": 4,
        "n_updates": 1464,
        "eval_freq": 15000,
        "hidden_dim_standard": 64,
        "hidden_dim_scaled": 45
    },
    "CP": {
        "env_type": "ComplexCartPole",
        "mt_setting": "None",
        "total_timesteps": 800000,
        "n_steps": 256,
        "num_envs": 4,
        "n_updates": 781,
        "eval_freq": 8000,
        "hidden_dim_standard": 64,
        "hidden_dim_scaled": 45
    },
    "IdenticalCP": {
        "env_type": "IdenticalCartPole",
        "mt_setting": "None",
        "total_timesteps": 800000,
        "n_steps": 256,
        "num_envs": 4,
        "n_updates": 781,
        "eval_freq": 8000,
        "hidden_dim_standard": 64,
        "hidden_dim_scaled": 45
    }
}

METHODS = [
    "varshare_standard",
    "varshare_scaled",
    "varshare_bayes",
    "varshare_partial",
    "varshare_prior_opt", # Prior as Hyperparam
    "soft_modularization",
    "paco",
    "shared_embedding",
    "shared_embedding_pcgrad"
]

def run_trial(trial, args):
    env_config = ENV_SETTINGS[args.env_key].copy()
    
    if args.test:
        print(">>> TEST MODE: Reducing timesteps and steps for verification")
        env_config["total_timesteps"] = 2000
        env_config["n_steps"] = 128
        env_config["n_updates"] = 5
        env_config["eval_freq"] = 1000
        env_config["num_envs"] = 2
    else:
        # Standard HPO Overrides (Cluster)
        if "HPO_TIME_STEPS" in os.environ:
             env_config["total_timesteps"] = int(os.environ["HPO_TIME_STEPS"])
             print(f">>> HPO_TIME_STEPS override: {env_config['total_timesteps']}")
             
        if "HPO_EVAL_FREQ" in os.environ:
             env_config["eval_freq"] = int(os.environ["HPO_EVAL_FREQ"])
    
    # --- Hyperparameters ---
    lr_actor = trial.suggest_float("lr_actor", 5e-5, 3e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-4, 5e-3, log=True)
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])
    
    # Common command construction
    # Initialize with shared args
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
        "--seed", str(trial.number)
    ]

    study_name = f"v4_{args.env_key}_{args.method}"
    if args.consistent_noise:
        study_name += "_consistent"
        
    # Determine if Baseline or VarShare
    is_baseline = args.method in ["soft_modularization", "paco", "shared_embedding", "shared_embedding_pcgrad"]
    
    if is_baseline:
        script = "scripts/train_baseline_ppo.py"
        cmd = cmd_base[:]
        cmd[1] = script
        cmd += ["--exp-name", f"{study_name}_trial{trial.number}"]
        
        # Method Specifics for Baseline
        hidden_dim = env_config["hidden_dim_standard"]
        cmd += ["--hidden-dim", str(hidden_dim)]
        
        if args.method == "soft_modularization":
            lr_routing = trial.suggest_float("lr_routing", 1e-4, 5e-3, log=True)
            cmd += [
                "--algo", "soft_mod",
                "--num-modules", "4",
                "--lr-routing", str(lr_routing)
            ]
        elif args.method == "paco":
            lr_weights = trial.suggest_float("lr_weights", 1e-4, 5e-3, log=True)
            cmd += [
                "--algo", "paco",
                "--num-experts", "4",
                "--lr-weights", str(lr_weights)
            ]
        elif args.method == "shared_embedding":
            cmd += ["--algo", "shared"]
        elif args.method == "shared_embedding_pcgrad":
            cmd += ["--algo", "pcgrad"]
            
    else:
        # VarShare Method
        script = "scripts/train_varshare_ppo.py"
        cmd = cmd_base[:]
        cmd[1] = script
        cmd += ["--exp-name", f"{study_name}_trial{trial.number}"]
        
        # Add VarShare Common Args
        if args.consistent_noise:
            cmd += ["--consistent-noise", "true"]
        else:
            cmd += ["--consistent-noise", "false"]
            
        # Method Specifics for VarShare
    
    # --- Method Specifics ---
    
    # Base Hidden Dim logic
    hidden_dim = env_config["hidden_dim_standard"]
    
    if args.method == "varshare_standard":
        rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
        kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
        cmd += [
            "--variant", "standard",
            "--hidden-dim", str(hidden_dim),
            "--rho-init", str(rho_init),
            "--kl-beta", str(kl_beta),
            "--embedding-type", "none"
        ]
        
    elif args.method == "varshare_scaled":
        rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
        kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
        hidden_dim = env_config["hidden_dim_scaled"]
        cmd += [
            "--variant", "standard",
            "--hidden-dim", str(hidden_dim),
            "--rho-init", str(rho_init),
            "--kl-beta", str(kl_beta),
            "--embedding-type", "none"
        ]
        
    elif args.method == "varshare_bayes":
        # Empirical Bayes
        # Usually we don't sweep KL Beta for Bayes? Or we do? 
        # Bayes usually implies KL Beta = 1 / NumTrainingSamples (handled internally or fixed).
        # But User's previous Bayes experiments often swept KL Beta too or set it to 1/N.
        # "VarShare Empirical Bayes" -> learned_prior=True.
        # Often we sweep kl_beta as a weighting factor still.
        rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
        kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
        cmd += [
            "--variant", "standard",
            "--hidden-dim", str(hidden_dim),
            "--rho-init", str(rho_init),
            "--kl-beta", str(kl_beta),
            "--learned-prior", "true",
            "--embedding-type", "none"
        ]
        
    elif args.method == "varshare_partial":
        rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
        kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
        cmd += [
            "--variant", "partial",
            "--hidden-dim", str(hidden_dim),
            "--rho-init", str(rho_init),
            "--kl-beta", str(kl_beta),
            "--embedding-type", "none"
        ]
        
    elif args.method == "varshare_prior_opt":
        # Prior Scale as Hyperparam
        rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
        kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
        prior_scale = trial.suggest_float("prior_scale", 1e-5, 1.0, log=True)
        cmd += [
            "--variant", "standard",
            "--hidden-dim", str(hidden_dim),
            "--rho-init", str(rho_init),
            "--kl-beta", str(kl_beta),
            "--prior-scale", str(prior_scale),
            "--embedding-type", "none"
        ]

    # Remove old dispatch logic which was appending indefinitely
    # And Baseline methods block logic is now handled above.

    # --- Execution ---
    # We parse the output to find the final evaluation score
    print(f"Running Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse "FINAL_EVAL_REWARD: X"
        final_reward = -float('inf')
        for line in output.split("\n"):
            if "FINAL_EVAL_REWARD:" in line:
                final_reward = float(line.split(":")[1].strip())
                
        if final_reward == -float('inf'):
            print("Warning: Could not parse Final Eval Reward. Returning -1000.")
            return -1000.0
            
        return final_reward
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"Stderr: {e.stderr}")
        return -1000.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", type=str, required=True, choices=ENV_SETTINGS.keys())
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("--consistent-noise", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--storage", type=str, default="sqlite:///v4_hpo.db")
    parser.add_argument("--test", action="store_true", help="Run in fast verification mode")
    args = parser.parse_args()
    
    study_name = f"v4_{args.env_key}_{args.method}"
    if args.consistent_noise:
        study_name += "_consistent"
        
    print(f"Starting HPO Study: {study_name}")
    print(f"Environment: {args.env_key}")
    print(f"Method: {args.method}")
    print(f"Consistent Noise: {args.consistent_noise}")
    
    print(f"Consistent Noise: {args.consistent_noise}")
    
    # Use hpo_utils for robust storage (Journal/RDB)
    sys.path.append("scripts")
    from hpo_utils import get_hpo_storage
    
    storage = get_hpo_storage(args.storage)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"Trials so far in DB: {len(study.trials)}")
    
    # Run exactly N_TRIALS in this process (Standard Worker Pattern)
    if args.n_trials > 0:
        study.optimize(lambda t: run_trial(t, args), n_trials=args.n_trials)
    
    print("Study Optimization Step Complete.")
    print(f"Best Params: {study.best_params}")
    try:
        print(f"Best Value: {study.best_value}")
    except ValueError:
        print("Best Value: None")

if __name__ == "__main__":
    main()
