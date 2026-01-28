
from run_overnight_utils import run_overnight_experiment

if __name__ == "__main__":
    # Algo 6: Scaled-Down VarShare
    # Matches "Active Parameter Count" of standard MLP (256 hidden).
    # VarShare Active Params = 2 * H^2 (Theta + Mu).
    # Standard Params = H^2.
    # To match: 2 * H_new^2 = 256^2  => H_new = 256 / sqrt(2) = 181.01 -> 181.
    
    run_overnight_experiment(
        algo_name="scaled_varshare",
        hidden_dim=181,
        varshare_args={
            "variant": "standard",
            "rho_init": -5.0,
            "mu_init": 0.0
        }
    )
