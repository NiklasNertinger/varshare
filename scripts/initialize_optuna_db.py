
import optuna
import os

# Define the storage URL
db_dir = os.path.expandvars("/netscratch/$USER/varshare/hpo_db")
os.makedirs(db_dir, exist_ok=True)
storage_url = f"sqlite:///{db_dir}/optuna_mt10.db"

print(f"Initializing Optuna Database at: {storage_url}")

# List of study names we expect to use
study_names = [
    "mt10_varshare",
    "mt10_shared",
    "mt10_paco",
    "mt10_soft_mod",
    "mt10_pcgrad",
    "mt10_oracle"
]

for study_name in study_names:
    print(f"Creating/Loading study: {study_name}")
    optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True
    )

print("Database initialization complete. Tables created.")
