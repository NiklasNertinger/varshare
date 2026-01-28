
import optuna
import os

from optuna.storages import JournalStorage, JournalFileStorage
import os

# Define the storage location
analysis_dir = os.path.expandvars("/netscratch/$USER/varshare/analysis")
os.makedirs(analysis_dir, exist_ok=True)
journal_path = os.path.join(analysis_dir, "optuna_journal.log")
storage = JournalStorage(JournalFileStorage(journal_path))

print(f"Initializing Optuna Journal at: {journal_path}")

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
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )

print("Database initialization complete. Tables created.")
