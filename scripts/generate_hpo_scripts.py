"""
Generates 11 separate HPO scripts based on the config in hpo_utils.py
"""
import os
from hpo_utils import STUDIES

TEMPLATE = """
import argparse
import optuna
import sys
import os


# Ensure we can import from scripts/
sys.path.append(os.getcwd())

from scripts.hpo_utils import run_trial, STUDIES
# Handle Optuna 4.0+ vs older
try:
    from optuna.storages import JournalStorage, JournalFileStorage
    # Check for v4 specific class if needed, or just use try/except block in storage init
    from optuna.storages.journal import JournalFileBackend
except ImportError:
    JournalFileBackend = None # Older optuna

STUDY_NAME = "{study_name}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-path", type=str, default="analysis/optuna_journal_mega.log")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    config = STUDIES[STUDY_NAME]
    
    # Storage Selection
    if args.storage_path.startswith("sqlite"):
        storage = optuna.storages.RDBStorage(args.storage_path)
    else:
        # Journal Storage (Cluster/NFS safe)
        if JournalFileBackend:
            # Optuna 4.0+
            storage = JournalStorage(JournalFileBackend(args.storage_path))
        else:
            # Legacy
            storage = JournalStorage(JournalFileStorage(args.storage_path))
            
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"Starting optimization for {{STUDY_NAME}}...")
    study.optimize(lambda t: run_trial(t, STUDY_NAME, config), n_trials=args.n_trials)
"""

def main():
    for study_name in STUDIES.keys():
        filename = f"scripts/optimize_{study_name}.py"
        content = TEMPLATE.format(study_name=study_name)
        
        with open(filename, "w") as f:
            f.write(content.strip())
            
        print(f"Generated {filename}")

if __name__ == "__main__":
    main()
