
from run_overnight_utils import run_overnight_experiment

if __name__ == "__main__":
    # Algo 3: VarShare + Reptile
    # Uses standard variant but enables the reptile update hook in training loop
    run_overnight_experiment(
        algo_name="varshare_reptile",
        use_reptile=True,
        varshare_args={
            "variant": "standard",
            "rho_init": -5.0,
            "mu_init": 0.0
        }
    )
