import optuna
import os
import argparse
from hpo_utils import get_hpo_storage

def clear_studies(analysis_dir):
    storage = get_hpo_storage(base_dir=analysis_dir)
    
    # List of study names we use
    study_names = [
        "mt10_varshare", 
        "mt10_shared", 
        "mt10_paco", 
        "mt10_soft_mod", 
        "mt10_pcgrad", 
        "mt10_oracle"
    ]
    
    print(f"Connecting to storage at: {analysis_dir}")
    
    for study_name in study_names:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"Successfully deleted study: {study_name}")
        except KeyError:
            print(f"Study '{study_name}' not found, skipping...")
        except Exception as e:
            print(f"Error deleting '{study_name}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=str, 
                        default=os.path.expandvars("/netscratch/$USER/varshare/analysis"),
                        help="Path to the analysis directory containing the journal log")
    args = parser.parse_args()
    
    # Confirm with user
    confirm = input(f"This will DELETE all HPO history in {args.analysis_dir}. Are you sure? (y/N): ")
    if confirm.lower() == 'y':
        clear_studies(args.analysis_dir)
        print("\nStudies cleared from Journal.")
        print("Note: You should now also run: rm -rf " + os.path.join(args.analysis_dir, "optuna/*"))
    else:
        print("Aborted.")
