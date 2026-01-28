"""
Unified Optimization Script for VarShare Mega-Batch (11 Studies).
Parses the study name argument and configures the Optuna search space accordingly.
"""
import sys
import os
import argparse
import subprocess
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

# Define Study Configs
STUDIES = {
    # --- Baseline & Architecture ---
    "mt10_varshare_base": {
        "variant": "standard", "embedding": "none", "args": {}
    },
    "mt10_varshare_emb_onehot": {
        "variant": "standard", "embedding": "onehot", "args": {} 
    },
    "mt10_varshare_emb_learned": { # Renamed from _learnable for consistency
        "variant": "standard", "embedding": "learned", "args": {}
    },
    "mt10_varshare_lora": {
        "variant": "lora", "embedding": "none", "args": {"lora-rank": 4}
    },
    "mt10_varshare_partial": { # Renamed from _last2
        "variant": "partial", "embedding": "none", "args": {}
    },
    "mt10_varshare_reptile": {
        "variant": "reptile", "embedding": "none", "args": {}
    },
    "mt10_varshare_scaled_down": {
        "variant": "standard", "embedding": "none", "args": {"hidden-dim": 181}
    },
    
    # --- Adaptive Methods ---
    "mt10_varshare_fixed_prior": {
        "variant": "standard", "embedding": "none", "args": {"prior-scale": 0.01} # Override search space? No, hardcode arg
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

def objective(trial, study_name, storage_url):
    config = STUDIES[study_name]
    
    # --- Standard Search Space (All Studies) ---
    lr_actor = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
    rho_init = trial.suggest_float("rho_init", -6.0, -2.0)
    kl_beta = trial.suggest_float("kl_beta", 1e-3, 1.0, log=True)
    # Categorical ent_coef
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.001, 0.005])
    
    # --- Conditional Search Space ---
    extra_cmd_args = []
    
    # Trigger Specific
    if "search_space" in config and "target_ratio" in config["search_space"]:
        target_ratio = trial.suggest_float("target_ratio", 0.05, 0.25)
        extra_cmd_args.extend(["--target-ratio", str(target_ratio)])
        
    # Emp Bayes Specific
    if "search_space" in config and "lambda_hyper" in config["search_space"]:
        lambda_hyper = trial.suggest_float("lambda_hyper", 1e-4, 1e-1, log=True)
        extra_cmd_args.extend(["--lambda-hyper", str(lambda_hyper)])
        
    # Fixed Prior Specific (Override standard search or logic?)
    # actually, fixed_prior just means we hardcode prior-scale.
    # The dictionary above sets 'prior-scale' arg.
    
    # --- Construct Command ---
    # Default Args for Scaled Architecture
    hidden_dim = config["args"].get("hidden-dim", 256)
    
    cmd = [
        "python", "scripts/train_varshare_ppo.py",
        "--algo", "varshare",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--total-timesteps", "2000000",
        "--num-envs", "8",
        "--n-steps", "2500", # 2.5k * 8 = 20k per update -> 100 updates total? No.
        # Check standard params: 
        # Usually n-steps=512, num-envs=4.
        # We want 2M steps.
        # Let's use sane defaults from previous HPO: n-steps=512, num-envs=8 ==> 4096 steps/batch.
        "--n-steps", "512",
        "--hidden-dim", str(hidden_dim),
        
        # HPO Params
        "--lr-actor", str(lr_actor),
        "--rho-init", str(rho_init),
        "--kl-beta", str(kl_beta),
        "--ent-coef", str(ent_coef),
        
        # Variant Config
        "--variant", config["variant"],
        "--embedding-type", config["embedding"],
    ]
    
    # Add Static Args from Config
    for k, v in config["args"].items():
        if k == "hidden-dim": continue # handled above
        # handle boolean flags
        if isinstance(v, bool):
             if v: cmd.append(f"--{k}") # Assumes store_true? No, we used type=lambda bool. So pass string 'true'
             cmd.append("true" if v else "false")
        else:
             cmd.extend([f"--{k}", str(v)])
             
    # Add Extra HPO Args
    cmd.extend(extra_cmd_args)
    
    # Logging
    run_name = f"{study_name}_trial_{trial.number}"
    cmd.extend(["--exp-name", run_name])
    
    print(f"Running Command: {' '.join(cmd)}")
    
    # Execute
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse Output for Success Rate
        for line in reversed(result.stdout.split('\n')):
             if "FINAL_EVAL_SCORE:" in line:
                 score = float(line.split(":")[-1].strip())
                 return score
        
        # Fallback to reading regular logs if needed, but explicit tag is safer
        return 0.0 # Fallback
        
    except subprocess.CalledProcessError as e:
        print(f"Trial failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage-path", type=str, default="analysis/optuna_journal_mega.log")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    # Use Journal Storage (Compatible with Cluster)
    # Check if local or cluster path logic needed? 
    # Just use the path provided.
    storage = JournalStorage(JournalFileStorage(args.storage_path))
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"Starting optimization for {args.study_name}...")
    study.optimize(lambda t: objective(t, args.study_name, args.storage_path), n_trials=args.n_trials)
