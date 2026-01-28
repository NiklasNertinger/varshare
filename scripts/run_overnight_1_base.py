
from run_overnight_utils import run_overnight_experiment

if __name__ == "__main__":
    # Algo 1: Base VarShare
    run_overnight_experiment(
        algo_name="base_varshare",
        varshare_args={
            "variant": "standard",
            "rho_init": -5.0,
            "mu_init": 0.0,
            "prior_scale": 1.0
        }
    )
