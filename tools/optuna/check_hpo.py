import sys
import os
import argparse
import optuna

def check_hpo(env_key, method="det_routing"):
    study_name = f"v5_{env_key}_{method}"
    storage_path = f"/netscratch/{os.environ.get('USER', 'nertinger')}/varshare/v5_hpo_{env_key}.log"
    
    if not os.path.exists(storage_path):
        print(f"Error: HPO log not found at {storage_path}")
        print("The array might not have started writing results yet.")
        return

    print(f"Loading Study: {study_name} from {storage_path}")
     # Ensure we can import hpo_utils
    try:
        from src.utils.hpo_utils import get_hpo_storage
        storage = get_hpo_storage(storage_path)
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"Failed to load optuna study: {e}")
        return

    # Extract all trials that have at least some value reported
    trials = [t for t in study.trials if t.value is not None]
    
    if not trials:
        print("No completed or reported trials found yet in this study.")
        print("Wait for the jobs to hit their first evaluation checkpoint.")
        return
        
    # Sort best first
    trials.sort(key=lambda t: t.value, reverse=True)
    
    print("\n" + "="*80)
    print(f" LIVE HPO STANDINGS: {env_key} | {method} ({len(trials)} trials recorded)")
    print("="*80)
    print(f"{'RANK':<6} | {'TRIAL_ID':<8} | {'REWARD':<10} | {'PARAMETERS'}")
    print("-" * 80)
    
    for i, t in enumerate(trials[:5]):
        # Format params nicely
        params_str = ", ".join([f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}" for k, v in t.params.items()])
        print(f"#{i+1:<5} | Trial {t.number:<4} | {t.value:<10.2f} | {params_str}")
        
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["CP", "LL", "MT4", "MT10"], help="Environment Key")
    parser.add_argument("--method", type=str, default="det_routing", help="Method to query")
    args = parser.parse_args()
    check_hpo(args.env, args.method)
