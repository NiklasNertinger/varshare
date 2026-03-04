import os
import json
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def get_best_hparams(storage_path, study_name_prefix):
    hparams = {}
    try:
        storage = JournalStorage(JournalFileStorage(storage_path))
    except (ImportError, AttributeError):
        # Fallback
        from optuna.storages.journal import JournalFileBackend
        storage = JournalStorage(JournalFileBackend(storage_path))

    studies = storage.get_all_studies()
    
    for study in studies:
        name = study.study_name
        if name.startswith(study_name_prefix):
            try:
                best_trial = storage.get_best_trial(study._study_id)
                # Parse out method from name, e.g., "v5_CP_det_base" -> "det_base"
                # "v4_MT4_varshare_standard_consistent" -> "varshare_standard_consistent"
                parts = name.split("_")
                # parts[0] is version, parts[1] is env
                method = "_".join(parts[2:])
                
                hparams[method] = best_trial.params
            except ValueError:
                # No trials finished
                print(f"No completed trials for study: {name}")
                
    return hparams

def main():
    base_dir = "context/HPOs"
    output_file = "context/HPOs/best_hparams.json"
    
    envs = ["CP", "LL", "MT4"]
    versions = ["v4", "v5"]
    
    all_best_hparams = {}
    
    for v in versions:
        all_best_hparams[v] = {}
        for env in envs:
            file_name = f"{v}_hpo_{env}.log"
            file_path = os.path.join(base_dir, file_name)
            
            if os.path.exists(file_path):
                print(f"Extracting carefully from {file_path}")
                prefix = f"{v}_{env}_"
                env_hparams = get_best_hparams(file_path, prefix)
                all_best_hparams[v][env] = env_hparams
            else:
                print(f"Warning: {file_path} not found.")

    # MT10 inherits MT4
    all_best_hparams["v4"]["MT10"] = all_best_hparams["v4"].get("MT4", {})
    all_best_hparams["v5"]["MT10"] = all_best_hparams["v5"].get("MT4", {})
    
    with open(output_file, 'w') as f:
        json.dump(all_best_hparams, f, indent=4)
        
    print(f"\nExtraction complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
