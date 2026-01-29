"""
Shared utilities for VarShare HPO.
"""
import subprocess
import optuna
import os
import sys

# Define Study Configs
STUDIES = {
    # --- Baseline & Architecture ---
    "mt10_varshare_base": {
        "variant": "standard", "embedding": "none", "args": {}
    },
    "mt10_varshare_emb_onehot": {
        "variant": "standard", "embedding": "onehot", "args": {} 
    },
    "mt10_varshare_emb_learned": { 
        "variant": "standard", "embedding": "learned", "args": {}
    },
    "mt10_varshare_lora": {
        "variant": "lora", "embedding": "none", "args": {"lora-rank": 4}
    },
    "mt10_varshare_partial": { 
        "variant": "partial", "embedding": "none", "args": {}
    },
    "mt10_varshare_reptile": {
        "variant": "reptile", "embedding": "none", "args": {}
    },
    "mt10_varshare_scaled_down": {
        "variant": "standard", "embedding": "none", "args": {"hidden-dim": 181}
    },
    
    # --- Clean Slate Tests (Depth) ---
    "mt10_varshare_base_400": {
        "variant": "standard", "embedding": "none", 
        "args": {"hidden-dim": 400, "num-layers": 3}
    },
    "mt10_varshare_base_64": {
        "variant": "standard", "embedding": "none", 
        "args": {"hidden-dim": 64, "num-layers": 2}
    },
    
    # --- Adaptive Methods ---
    "mt10_varshare_fixed_prior": {
        "variant": "standard", "embedding": "none", "args": {"prior-scale": 0.01} 
    },
    "mt10_varshare_annealing": {
        "variant": "standard", "embedding": "none", 
        "args": {"kl-schedule": "annealing", "warmup-frac": 0.1}
    },
    "mt10_varshare_trigger": {
        "variant": "standard", "embedding": "none", 
        "args": {"kl-schedule": "trigger"},
        "search_space": ["target_ratio"]
    },
    "mt10_varshare_emp_bayes": {
        "variant": "standard", "embedding": "none", 
        "args": {"learned-prior": True},
        "search_space": ["lambda_hyper"]
    }
}


def run_trial(trial, study_name, config, n_envs=8, n_steps=512):
    # --- Standard Search Space ---
    lr_actor = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
    rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
    kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])
    
    # --- Extra Search Space (Conditional) ---
    extra_cmd_args = []
    
    if "search_space" in config:
        if "target_ratio" in config["search_space"]:
            target_ratio = trial.suggest_float("target_ratio", 0.05, 0.25)
            extra_cmd_args.extend(["--target-ratio", str(target_ratio)])
            
        if "lambda_hyper" in config["search_space"]:
            lambda_hyper = trial.suggest_float("lambda_hyper", 1e-4, 1e-1, log=True)
            extra_cmd_args.extend(["--lambda-hyper", str(lambda_hyper)])

    # --- Construct Command ---
    hidden_dim = config["args"].get("hidden-dim", 256)
    
    # Allow Environment Overrides for Testing (e.g. 30k steps)
    total_timesteps = os.environ.get("HPO_TIME_STEPS", "2000000")
    n_steps_arg = os.environ.get("HPO_N_STEPS", str(n_steps))
    
    cmd = [
        sys.executable, "scripts/train_varshare_ppo.py",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", str(total_timesteps),
        "--num-envs", str(n_envs),
        "--n-steps", str(n_steps_arg),
        "--hidden-dim", str(hidden_dim),
        
        # HPO
        "--lr-actor", str(lr_actor),
        "--rho-init", str(rho_init),
        "--kl-beta", str(kl_beta),
        "--ent-coef", str(ent_coef),
        
        # Variants
        "--variant", config["variant"],
        "--embedding-type", config["embedding"],
    ]
    
    # Static Args
    for k, v in config["args"].items():
        if k == "hidden-dim": continue
        if isinstance(v, bool):
             if v: cmd.append(f"--{k}") # Flag
             cmd.append("true" if v else "false")
        else:
             cmd.extend([f"--{k}", str(v)])
             
    cmd.extend(extra_cmd_args)
    
    run_name = f"{study_name}_trial_{trial.number}"
    cmd.extend(["--exp-name", run_name])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse for Reward
        score = -9999.0
        for line in reversed(result.stdout.split('\n')):
             if "FINAL_EVAL_REWARD:" in line:
                 score = float(line.split(":")[-1].strip())
                 return score
                 
        # Fallback (Success Rate parse)
        # Maybe log warning?
        return 0.0
        
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return -9999.0

def get_trial_params(trial, algo):
    """
    Standard parameter ranges for defined algorithms.
    """
    params = {}
    
    # Common PPO Hyperparams
    params["learning_rate_actor"] = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
    if algo == "varshare":
         # VarShare typically uses same or similar LR for critic, or specific
         params["learning_rate_critic"] = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
         params["rho_init"] = trial.suggest_float("rho_init", -6.0, -2.0)
         params["kl_beta"] = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
         params["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])
    
    elif algo == "shared":
         params["learning_rate_actor"] = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
         params["learning_rate_critic"] = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
         params["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])

    elif algo == "pcgrad":
         params["learning_rate_actor"] = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
         params["learning_rate_critic"] = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
         params["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])

    elif algo == "paco":
         params["learning_rate_actor"] = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
         params["learning_rate_critic"] = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
         params["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])
         params["num_experts"] = trial.suggest_int("num_experts", 2, 8)
         params["lr_weights"] = trial.suggest_float("lr_weights", 1e-4, 1e-2, log=True)

    elif algo == "soft_mod":
         params["learning_rate_actor"] = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
         params["learning_rate_critic"] = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
         params["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])
         params["num_modules"] = trial.suggest_int("num_modules", 2, 8)
         params["lr_routing"] = trial.suggest_float("lr_routing", 1e-4, 1e-2, log=True)
         
    elif algo == "oracle":
         params["learning_rate_actor"] = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
         params["learning_rate_critic"] = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
         params["ent_coef"] = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])

    return params

def calculate_objective(history):
    """
    Calculate scalar objective from training history.
    """
    if not history:
        return -9999.0
    
    # Get last entry
    last_entry = history[-1]
    
    # Prioritize Eval Reward
    if "eval_reward" in last_entry and last_entry["eval_reward"] is not None:
        return last_entry["eval_reward"]
    
    # Fallback to last mean reward
    if "reward" in last_entry:
        return last_entry["reward"]
        
    return -9999.0

def get_hpo_storage(storage_path):
    """
    Return Optuna storage object (Journal or RDB).
    """
    if storage_path.startswith("sqlite"):
        return optuna.storages.RDBStorage(storage_path)
    
    try:
        from optuna.storages import JournalStorage, JournalFileStorage
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(storage_path))
    except (ImportError, AttributeError):
        # Fallback for older Optuna
        from optuna.storages import JournalStorage, JournalFileStorage
        return JournalStorage(JournalFileStorage(storage_path))
