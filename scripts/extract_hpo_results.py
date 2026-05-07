import os
import json
import optuna

def main():
    user = os.environ.get("USER", "nertinger")
    db_file = f"/netscratch/{user}/varshare/test_hpo.db"
    storage_uri = f"sqlite:///{db_file}"
    output_file = "context/HPOs/best_hparams.json"
    
    print(f"Connecting to Optuna database: {storage_uri}")
    if not os.path.exists(db_file):
        print(f"Error: Database file {db_file} does not exist yet.")
        return

    # Initialize optuna storage
    storage = optuna.storages.RDBStorage(url=storage_uri, engine_kwargs={"timeout": 60})
    studies = storage.get_all_studies()
    
    # Mapping from HPO study method names to the exact keys used in submit_final_campaign.py
    method_mapping = {
        "shared": "shared_embedding",
        "independent": "independent",
        "pcgrad": "shared_embedding_pcgrad",
        "cagrad": "shared_embedding_cagrad",
        "paco": "paco",
        "soft_mod": "soft_mod",
        "moore": "moore",
        "base": "det_base",
        "routing": "det_routing"
    }
    
    hparams_data = {
        "v5": {
            "MT10": {}
        }
    }
    
    for study in studies:
        name = study.study_name
        # Expecting study names like: hpo_MT10_routing
        if name.startswith("hpo_MT10_"):
            method = name.replace("hpo_MT10_", "")
            if method in method_mapping:
                target_key = method_mapping[method]
                try:
                    best_trial = storage.get_best_trial(study._study_id)
                    hparams_data["v5"]["MT10"][target_key] = best_trial.params
                    print(f"  -> Successfully extracted champion for '{target_key}' with score: {best_trial.value:.2f}")
                except ValueError:
                    print(f"  -> No completed trials for study: {name}. Skipping.")
    
    # Save the output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(hparams_data, f, indent=4)
        
    print(f"\nChampion extraction complete! Saved best hyperparameters to: {output_file}")

if __name__ == "__main__":
    main()
