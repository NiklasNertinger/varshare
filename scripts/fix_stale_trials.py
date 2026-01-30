import optuna
import os
import argparse
from hpo_utils import get_hpo_storage

def fix_stale_trials(journal_path):
    if not os.path.exists(journal_path):
        print(f"File not found: {journal_path}")
        return

    storage = get_hpo_storage(journal_path)
    
    # Get all studies
    studies = optuna.get_all_study_summaries(storage=storage)
    
    for study_sum in studies:
        try:
            study = optuna.load_study(study_name=study_sum.study_name, storage=storage)
            stale_count = 0
            # iterate snapshot
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    print(f"Failing stale trial {trial.number} in study '{study_sum.study_name}'...")
                    try:
                        # Use study.tell() to update state safely
                        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                        stale_count += 1
                    except Exception as e:
                        print(f"Could not fail trial {trial.number}: {e}")
            
            if stale_count > 0:
                print(f"Fixed {stale_count} stale trials in study '{study_sum.study_name}'")
        except Exception as e:
            print(f"Error processing study {study_sum.study_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=str, required=True)
    args = parser.parse_args()
    
    fix_stale_trials(args.journal)
