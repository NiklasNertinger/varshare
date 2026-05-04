# VarShare Repository Structure

This document codifies the strictly flattened architecture of the VarShare repository. Any AI agent modifying or expanding this codebase must preserve this separation of concerns.

## 1. Execution Entry Points (Root Level)
All training loops and primary execution files live at the root of the repository.
- `train_baselines.py`: The master execution loop for standard literature baselines (PaCo, Soft Modularization, PCGrad, etc.).
- `train_routing.py`: The master execution loop for our novel Deterministic Gradient Routing architecture.

## 2. Core Logic (`src/`)
All mathematically intensive logic, PyTorch architectures, and environments are decoupled from the execution loops and stored in `src/`.
- `src/models_baselines.py`: Contains the raw PyTorch classes for all baselines.
- `src/models_routing.py`: Contains the PyTorch classes and explicit $L_2$-regularized residual adapters for the active routing architecture.
- `src/algo/`: Contains the gradient surgery mathematical logic (e.g., `pcgrad.py`, `cagrad.py`).
- `src/env/`: Contains the MetaWorld wrappers and identical/complex CartPole physics generators.

## 3. Orchestration & Tooling (`scripts/`)
The `scripts/` directory is strictly reserved for cluster orchestration, hyperparameter optimization logic, and data analysis.
- `scripts/optimize_v5.py`: SLURM HPO sweep orchestrator.
- `scripts/submit_final_campaign.py`: Final phase benchmarking orchestrator.
- `scripts/monitor_runs.py`: Parses the `/netscratch/` storage to track real-time training progress.
- `scripts/plot_final_campaign.py`: Generates all scientific matplotlib figures from the resultant tensorboard data.
- **Bash Wrappers:** All `.sh` files (e.g., `run_v5_worker.sh`, `submit_routing_pipeline.sh`) are used to interface with SLURM.

## 4. Temporary / Local Dumps
- `analysis/`: Auto-generated matplotlib outputs.
- `logs/` & `runs/`: TensorBoard dumps.
*(Note: On the cluster, these paths are dynamically overridden to target `/netscratch/$USER/varshare/` to avoid home-directory bloat).*
