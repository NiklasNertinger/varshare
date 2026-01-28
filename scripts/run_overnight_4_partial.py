
from run_overnight_utils import run_overnight_experiment

if __name__ == "__main__":
    # Algo 4: Partial VarShare
    # First layers are Standard (Shared), last 2 layers are VarShare.
    run_overnight_experiment(
        algo_name="partial_varshare",
        varshare_args={
            "variant": "partial",
            "rho_init": -5.0,
            "mu_init": 0.0
        }
    )
