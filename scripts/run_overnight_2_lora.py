
from run_overnight_utils import run_overnight_experiment

if __name__ == "__main__":
    # Algo 2: VarShare + LoMA
    run_overnight_experiment(
        algo_name="varshare_lora",
        varshare_args={
            "variant": "lora",
            "rank": 4, # Low rank for parameter efficiency
            "rho_init": -5.0,
            "prior_scale": 1.0
        }
    )
