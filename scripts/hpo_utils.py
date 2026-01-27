import optuna
import os
import torch
import numpy as np

# Common Hyperparameter Ranges
COMMON_SEARCH_SPACE = {
    "lr_actor": {"low": 1e-4, "high": 3e-3, "log": True},
    "lr_critic": {"low": 3e-4, "high": 3e-3, "log": True},
    "ent_coef": {"choices": [0.0, 0.001, 0.005, 0.01]},
}

VARSHARE_SEARCH_SPACE = {
    "kl_beta": {"low": 1e-4, "high": 1.0, "log": True},
    "rho_init": {"low": -7.0, "high": -3.0},
}

PACO_SEARCH_SPACE = {
    "lr_weights": {"low": 1e-4, "high": 1e-2, "log": True},
}

SOFTMOD_SEARCH_SPACE = {
    "lr_routing": {"low": 1e-4, "high": 1e-2, "log": True},
}

def get_hpo_storage():
    """Returns the shared SQLite storage string."""
    db_path = os.path.join(os.getcwd(), "optuna_mt10.db")
    return f"sqlite:///{db_path}"

def get_trial_params(trial, algo):
    """Samples parameters for a given trial and algorithm."""
    params = {}
    
    # Common PPO Params
    params["learning_rate_actor"] = trial.suggest_float("lr_actor", **COMMON_SEARCH_SPACE["lr_actor"])
    params["learning_rate_critic"] = trial.suggest_float("lr_critic", **COMMON_SEARCH_SPACE["lr_critic"])
    params["ent_coef"] = trial.suggest_categorical("ent_coef", COMMON_SEARCH_SPACE["ent_coef"]["choices"])
    
    # Algo Specifics
    if algo == "varshare":
        params["kl_beta"] = trial.suggest_float("kl_beta", **VARSHARE_SEARCH_SPACE["kl_beta"])
        params["rho_init"] = trial.suggest_float("rho_init", **VARSHARE_SEARCH_SPACE["rho_init"])
        params["prior_scale"] = 1.0 # Fixed
        
    elif algo == "paco":
        params["lr_weights"] = trial.suggest_float("lr_weights", **PACO_SEARCH_SPACE["lr_weights"])
        
    elif algo == "soft_mod":
        params["lr_routing"] = trial.suggest_float("lr_routing", **SOFTMOD_SEARCH_SPACE["lr_routing"])
        
    return params

def calculate_objective(history):
    """
    Calculates the objective value (scalar) from training history.
    Strategy: Average Eval Success Rate over the last 3 evaluations.
    """
    if not history:
        return 0.0
        
    # Filter for entries that have eval_success data
    eval_entries = [h for h in history if h.get("eval_success") is not None]
    
    if not eval_entries:
        # Fallback to train success if no eval happened (should not happen if budget is sufficient)
        return history[-1].get("success", 0.0)
        
    # Take last 3 evals
    last_n = min(3, len(eval_entries))
    recent_evals = eval_entries[-last_n:]
    
    avg_success = np.mean([e["eval_success"] for e in recent_evals])
    return avg_success
