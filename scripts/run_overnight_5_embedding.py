
from run_overnight_utils import run_overnight_experiment

if __name__ == "__main__":
    # Algo 5: Task Embedding (Baseline)
    # Standard MLP with Task ID embedding concatenated to input.
    # Disables VarShare.
    run_overnight_experiment(
        algo_name="embedding", # Triggers use_varshare=False in utils
        use_task_embedding=True,
        varshare_args={}
    )
