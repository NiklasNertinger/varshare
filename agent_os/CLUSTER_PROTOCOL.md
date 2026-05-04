# VarShare Cluster Protocol (`pegasus.dfki.de`)

This file codifies the explicit cluster hygiene and orchestration etiquette required for the VarShare project. Any AI agent operating in this repository must strictly adhere to these rules.

## Connection & Environment
1. **Hostname:** `pegasus.dfki.de`
2. **Environment:** All Python execution must occur within the `venv` virtual environment located at `~/.venv` (or equivalent local `.venv` on Windows).
3. **Paths:** 
   - Code lives in `~/varshare` (or local equivalent).
   - Heavy artifacts (logs, models, W&B local data) MUST be saved to `/netscratch/$USER/varshare/`. Do not pollute the home directory.

## SLURM Hygiene
1. **Hard Limits:** The cluster enforces a strict 72-hour wall-clock limit (`--time=72:00:00`).
2. **Partition:** Submit all jobs to the `batch` partition.
3. **Array Jobs:** When performing Hyperparameter Optimization (HPO), you must use SLURM Job Arrays (e.g., `--array=1-25%8`) to parallelize trials instead of running sequential loops inside a single job.
4. **Dependency Chaining:** Use `#SBATCH --dependency=afterok:<job_id>` to chain the evaluation phase directly after the HPO array finishes.

## Logging & Observability
1. **Weights & Biases:** We use `wandb` for all interactive monitoring. The entity is `niklas-nertinger-university-of-oxford` and the project is `varshare`.
2. **Offline Resilience:** Since compute nodes may occasionally lose network access, `wandb` logic must not crash the training loop. Use offline syncing (`WANDB_MODE=offline`) or `try/except` wrappers.
3. **Axis Synchronization:** All metrics (training and evaluation) must be forcefully logged using a unified `global_step` to prevent disjointed graphs in the W&B dashboard.
