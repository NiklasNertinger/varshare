# VarShare Project: Comprehensive Technical Handover

**Version:** 1.0 (Post-Advanced Metrics Implementation)
**Date:** February 17, 2026

---

# 1. Executive Technical Overview

## 1.1 Project Mission
This repository implements **Variational Parameter Sharing (VarShare)**, a novel neural architecture for Multi-Task Reinforcement Learning (MTRL). The core hypothesis is that learning a probabilistic "shared backbone" ($\theta$) with task-specific variational adapters ($\mu_m, \sigma_m$) allows for:
1.  **Implicit Curriculum:** Tasks can lean on shared knowledge ($\theta$) early in training.
2.  **Conflict Resolution:** Task-specific adapters absorb gradient conflicts that would otherwise destabilize the shared backbone.
3.  **Automatic Regularization:** The KL-divergence penalty acts as a "Information Bottleneck," encouraging tasks to share parameters unless specialization provides significant reward gain.

## 1.2 Current Stability & Status
*   **Core Architecture (`src/models.py`):** STABLE via `VarShareNetwork` (v3). Verified to support Standard, LoRA, Partial, and Reptile variants.
*   **Algorithm (`src/algo/ppo.py`):** STABLE via PPO with GAE-Lambda and Clipped Surrogate Objective. Includes custom KL-penalty logic and Empirical Bayes support.
*   **Baselines (HPO V3):** STABLE. The configuration bug preventing MT4 runs has been patched. The "Restart" scripts (`run_production_restart_baselines_v3.sh`) are active.
*   **Experiments (Mega HPO):** IN-PROGRESS. A massive hyperparameter optimization campaign for 11 variants is currently paused or running on the cluster. **Data preservation is critical.**

## 1.3 Objectives & Roadmap
*   **Verification:** Validate VarShare's superiority over PaCo (Parameter Composition), Soft Modularization, and PCGrad on Meta-World MT10 and MT50 benchmarks.
*   **Measurement:** Quantify "Sparsity of Specialization" and "Residual SNR" to prove the mechanism of action.
*   **Publication:** This code is the primary artifact for an upcoming conference submission.

---

# 2. Repository Architecture & Anatomy

## 2.1 File Structure Deep-Dive
```
.
├── analysis/               # [ARTIFACTS] Root directory for all experiment outputs.
│   ├── mega_hpo_v3/        # [CRITICAL] Data for the main VarShare parameter study. DO NOT DELETE.
│   ├── scaled_v3/          # [CRITICAL] Data for the Baselines restart.
│   └── test_metrics_plots/ # [TEMP] Verification run for plotting logic.
├── context/                # [DOCS] LLM Context and Project Specs.
├── logs/                   # [CLUSTER] Slurm stdout/stderr. Watch here for OOM errors.
├── scripts/                # [EXECUTABLE] Entry points.
│   ├── optimize_*.py       # [HPO] Optuna wrapper scripts.
│   ├── submit_*.sh         # [SLURM] Job submission descriptors.
│   ├── run_*.sh            # [ORCHESTRATION] Local wrappers to launch Slurm jobs.
│   ├── train_varshare_ppo.py # [CORE] Main training loop for VarShare.
│   └── train_baseline_ppo.py # [CORE] Main training loop for Baselines.
├── src/                    # [LIBRARY]
│   ├── algo/               # RL Algorithms.
│   │   ├── ppo.py          # Proximal Policy Optimization implementation.
│   │   └── buffers.py      # (Implicit) Rollout storage logic.
│   ├── env/                # Environment Wrappers.
│   │   ├── metaworld_wrapper.py # MT10/MT50 interface.
│   │   └── cartpole_wrapper.py  # Debug environments.
│   └── models.py           # Neural definitions (VarShareLayer, PaCo, etc.).
└── wandb/                  # [LOGGING] Local W&B cache (Disabled/Optional).
```

## 2.2 Key File Interactions
1.  **Orchestration:** `run_production_restart_baselines_v3.sh` sets env vars (`HPO_MT_SETTING`, `HPO_N_TRIALS`) -> Calls `submit_restart_baselines_v3.sh`.
2.  **Scheduling:** `submit_restart_baselines_v3.sh` creates a Slurm Job Array (0-29) -> Calls `optimize_mt10_*.py` for each task ID.
3.  **Optimization:** `optimize_mt10_pcgrad_scaled.py` (e.g.) uses Optuna to sample hyperparameters -> Builds a command string -> Calls `train_baseline_ppo.py`.
4.  **Training:** `train_baseline_ppo.py` instantiates `src.env.MetaWorldWrapper` and `src.models.ActorCritic` -> Runs PPO loop -> Saves to `analysis/`.

---

# 3. Canonical vs Outdated Files

## 3.1 Canonical Implementations (The "Golden" Set)
*   **VarShare Model:** `src/models.py` :: `VarShareLayer` and `VarShareNetwork`.
*   **Baseline Models:** `src/models.py` :: `PaCoActorCritic`, `SoftModularActorCritic`.
*   **Training Loop (VarShare):** `scripts/train_varshare_ppo.py`.
*   **Training Loop (Baselines):** `scripts/train_baseline_ppo.py`.
*   **Algorithmic Update:** `src/algo/ppo.py`.

## 3.2 Deprecated / Legacy Files
*   `scripts/optimize_mt10_*.py` (without `_scaled` suffix): These target the older "Small" architecture ([64, 64]). The V3 experiments use "Scaled" ([256, 256]).
*   `scripts/submit_restart_baselines.sh` (No `_v3`): Replaced by `_v3` version to fix the MT4 argument bug.

## 3.3 Shadow Implementations
*   `src/algo/meta_learning.py` (MAML/Reptile logic): Currently unused in the main PPO pipeline. Exists for future "Meta-RL vs Multi-Task RL" comparisons.

---

# 4. File Organization Philosophy

*   **"One Script per Role":** We avoid a single `main.py` with 1000 arguments. Instead, we have `train_varshare_ppo.py` vs `train_baseline_ppo.py` to keep logic distinct.
*   **"Code as Configuration":** Architectural variants (LoRA, Reptile, Partial) are handled via string arguments in `models.py` rather than separate classes, to maximize code sharing in the `forward` pass.
*   **Metric Centralization:** All architectural metrics are computed effectively "in-graph" via `get_architectural_metrics` methods on the layers, then aggregated by the network. This ensures metrics stay in sync with the model definition.

---

# 5. Experiment Pipeline Architecture

## 5.1 The HPO Hierarchy
The project relies on **Optuna** for hyperparameter tuning.
1.  **Storage:** SQLite DBs located at `/netscratch/$USER/varshare/analysis/<exp_group>/optuna_journal.db`.
2.  **Samplers:** TPE (Tree-structured Parzen Estimator) is used to sample Learning Rates, PPO Clip parameters, and Entropy Coefficients.
3.  **Pruning:** Current scripts (V3) use `MedianPruner` to kill trials that underperform the median at 25% of training steps.

## 5.2 Determinism & Seeding
*   **Seeding:** Explicitly set for Python, Numpy, and Torch (`torch.manual_seed`).
*   **Cuda Determinism:** `torch.backends.cudnn.deterministic = True` is set, but some atomic reductions on GPU (like `torch.index_add_` used in scatter operations) may still introduce micro-variance.
*   **Env Seeding:** Meta-World environments are seeded via `env.seed(seed)`.

## 5.3 Data Flow
1.  **Config:** `args` parsed in `train_*.py`.
2.  **Runtime:** `heartbeat.csv` is appended every update cycle (~2048 steps).
3.  **Synthesis:** At end of training, `history.json` is dumped (full run data).
4.  **Plotting:** Matplotlib generates PNGs from `history.json` immediately.

---

# 6. Cluster & Compute Infrastructure

## 6.1 Slurm Configuration
*   **Partition:** `gpu_std` (Standard GPU nodes).
*   **Time Limit:** 
    *   Test: 20-30 mins.
    *   Production: 6-24 hours.
*   **Resources:**
    *   `--gres=gpu:1` (Usually A100 or V100).
    *   `--cpus-per-task=8` (Needed for Meta-World's heavy MuJoCo physics).
    *   `--mem=32G` (Meta-World MT50 is memory hungry).

## 6.2 File System Constraints
*   **Home Dir:** (`/home/$USER`) Small quota. Code lives here.
*   **Netscratch:** (`/netscratch/$USER`) Large quota, fast I/O. **ALL DATA MUST GO HERE.**
*   **Venv:** The virtual environment is assumed to be at `/netscratch/$USER/varshare/venv` for cluster jobs.

## 6.3 Resume Logic
The current HPO scripts check for "Completed Trials" in the Optuna DB. If a job is killed and restarted:
1.  Optuna sees the trial state. (If FAIL/PRUNED/COMPLETE, it starts a new one).
2.  Checkpointing within a trial (resuming from step X) is **NOT** currently implemented. If a job dies mid-trial, the trial is lost/failed.

---

# 7. Metrics & Evaluation Contracts

## 7.1 Performance Metrics
*   **Eval Reward:** The primary success metric. computed periodically (every 25k steps). Deterministic policy ($\pi_{det} = \arg\max \pi$).
*   **Success Rate:** Binary (1/0) per episode. Averaged.

## 7.2 Structural Metrics (The "Novelty")
These metrics are the scientific core of the project.
*   **Sparsity:** Portion of task-specific parameters ($\mu$) that are effectively zero ($< 1\%$ of backbone norm).
    *   *Hypothesis:* VarShare should yield high sparsity (efficient use of capacity).
*   **Residual SNR:** Signal-to-Noise Ratio ($|\mu| / \sigma$) of the adapters.
    *   *High SNR* = Confident Specialization.
    *   *Low SNR* = Posterior Collapse / Unused capacity.
*   **Sharing Ratio:** $R = \frac{\|\theta\|}{\|\theta\| + \|\mu\|}$.
    *   *Hypothesis:* VarShare maintains $R > 0.8$, whereas naive finetuning drops to $R < 0.5$.

## 7.3 Optimization Metrics
*   **KL Penalty (Scaled):** $\beta \cdot \text{KL}(q || p)$. The actual term added to PPO Loss.
*   **Raw KL:** The information-theoretic distance (in Nats) between the adapter posterior and the prior.
*   **Explained Variance:** $1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$. Diagnosis of Value Function quality.

---

# 8. Logging & Artifact Tracking

## 8.1 Directory Layout (Analysis)
```
analysis/
  <Experiment_Name>/
    seed_1/
      heartbeat.csv       # Row-per-update log
      history.json        # Full run dictionary (JSON)
      model_final.pt      # PyTorch State Dict (CPU mapped)
      *.png              # Generated Plots
      summary.txt         # Text report
```

## 8.2 Reproducibility Risk
*   **Naming Collisions:** If you run `train_varshare_ppo.py` with the same `--exp-name` and `--seed` twice, it **WILL OVERWRITE** the previous run without warning (unless modifying `train.py` logic which currently does `os.makedirs(..., exist_ok=True)`).
*   **Code Versioning:** Results in `analysis/` are not automatically linked to a git commit. You must ensure you are on the right commit matching the logs.

---

# 9. Configuration Management

## 9.1 Environment Variables
*   `HPO_MT_SETTING`: Controls the benchmark (MT10, MT50, MT4).
*   `HPO_N_TRIALS`: Controls loop count in Submit scripts.
*   `HPO_TIME_STEPS`: Controls total training duration.
*   `HPO_EVAL_FREQ`: Controls evaluation density.

## 9.2 Hardcoded Constants
*   `train_varshare_ppo.py`:
    *   `print_freq = 2500` steps for console logging.
    *   `reward_window = deque(maxlen=25)` for smoothing.
*   `src/models.py`:
    *   `epsilon = 1e-8` for numerical stability in division.
    *   `mu_init (default) = 0.0`.
    *   `rho_init (default) = -5.0` (yields sigma $\approx 0.006$).

---

# 10. Dependency Graph & Coupling

*   **PPO <-> Models:** `PPO` class calls `agent.get_kl()`. It assumes the agent has this method. If you switch to a standard `ActorCritic` without `get_kl`, PPO handles it (returns 0), but `train_varshare_ppo.py` logic relies on `kl_beta` being irrelevant.
*   **Train <-> Metrics:** `train_varshare_ppo.py` explicitly scrapes keys like `norm_mu_layer_0` from the dictionary returned by `agent.get_architectural_metrics()`. If you rename keys in `models.py`, plotting logic in `train_varshare_ppo.py` will fail or plot nothing.

---

# 11. Known Technical Debt

1.  **JSON Serialization:** `train_varshare_ppo.py` now includes a custom `NumpyEncoder` class to handle `float32`. This should be moved to a util file (`src/utils.py`) if reused.
2.  **No Checkpoint Resume:** As mentioned, cluster jobs cannot resume mid-training. This wastes compute if a 12-hour job dies at hour 11.
3.  **Plotting Memory Usage:** `history.json` is loaded fully into RAM for plotting. For 10M+ steps, this JSON could be 500MB+, causing slow plotting or OOM on weak nodes. Streaming plotting would be better.

---

# 12. Reproducibility Guarantees & Gaps

*   **Verified:**
    *   Single-seed runs on the same hardware produce identical `heartbeat.csv` rows.
    *   Metric Calculation logic is unit-tested via `verify_metrics_50k.ps1`.
*   **Gaps:**
    *   Cross-Platform accumulation (Linux Cluster vs Windows Local) often differs slightly due to floating point precision and OS-specific threading. DO NOT mix OS results in strict comparisons.

---

# 13. Data Management

*   **Raw Data:** `heartbeat.csv` is the most robust source.
*   **Derived Data:** `history.json` contains the same data but structured.
*   **Cleanup:** `scripts/run_production_*.sh` scripts often include `rm -rf` commands to clean "Scaled" directories. **BE EXTREMELY CAREFUL** running these scripts manually.

---

# 14. Implicit Invariants

1.  **Obs Normalization:** The current V3 pipeline **DOES NOT** use `VecNormalize` or `NormalizeObservation`.
    *   *Observation:* Training runs on raw MuJoCo observations. This is sub-optimal for PPO but ensures consistent comparison across all V3 variants currently running.
    *   *Warning:* Do NOT add normalization now without restarting the entire Mega HPO campaign, as it would shift the input distribution and invalidate comparisons.
    *   *Inference:* `model_final.pt` contains only weights. Since no normalization stats exist, loading the model for inference is straightforward (no missing `obs_rms` stats).

2.  **Reward Scaling:** Rewards are NOT normalized/clipped in PPO. Value targets can be large (e.g. 5000+ for MT10). PPO Value Clipping uses `clip_coef` which assumes normalized values near 1.0? No, typically Value Clipping is relative to previous value. This project uses `0.5 * v_loss_max` logic.

---

# 15. Baseline Sanctity

The following Baselines are contractually defined for the paper:
1.  **Shared PPO (STL/MTL):** Standard PPO with no task ID or just task embedding.
2.  **PCGrad:** Gradient projection method.
3.  **PaCo:** Parameter Composition (4 experts).
4.  **SoftMod:** Soft Modularization (Routing).

**Constraint:** The hyperparameters for these baselines (LR, network size) MUST MATCH the VarShare "Scaled" setup ([256, 256] backbone) to be a fair comparison. Do not sneakily optimize baselines with different settings.

---

# 16. Architectural Decisions That May Look Questionable

*   **"Partial" Variant:** Why only the last layer?
    *   *Reasoning:* Early layers in CNNs/MLPs extract generic features. Specialization is most valuable near the output (policy/value heads). Full-network VarShare wastes KL budget on "boring" early layers.
*   **Learned Prior (Empirical Bayes):** Why optimize the prior?
    *   *Reasoning:* A fixed prior (sigma=0.1) might be too loose or too strict. Learning the prior allows the network to say "I don't need specialization here, tighten the prior to 0" (Sparsity).

---

# 17. Things Previously Tried That Did Not Work

*   **Naive MT4 Launch:** Failed because `train_baseline_ppo.py` had a hardcoded list `["MT10", "MT50"]` and rejected "MT4". Detailed in the "Walkthrough - Fixing V3 Baseline Failures".
*   **Full-Rank Adapters:** Initial experiments showed massive instability. LoRA (Rank 4) stabilized the variational updates significantly.

---

# 18. Catastrophic Failure Modes

1.  **Loss of Normalization Stats:** As noted in Section 14, we are not saving `VecNormalize` stats. If you delete `analysis/` and try to re-render videos later from `model_final.pt`, the agent will act randomly because it expects normalized inputs.
2.  **Database Locking:** Optuna SQLite DBs on network file systems (`/netscratch`) can sometimes lock (`database is locked`). The scripts use `flock` or reliance on Optuna's internal retry, but file-system latency can cause crashes.

---

# 19. Safety Boundaries for Future Modifications

*   **Algorithm:** Do not touch `src/algo/ppo.py`'s `update` function without re-running `verify_metrics_50k.ps1`. A sign error in the KL term can invert the regularization (encouraging divergence instead of penalizing it).
*   **Architecture:** Changing `VarShareLayer` default `prior_scale` changes the implicit regularization strength of ALL running experiments. Version bumping is required for such changes.

---

# 20. Minimal Knowledge Required Before Modifying Core Logic

1.  **Variational Inference:** Understand that we serve `mu + sigma * epsilon` during training, but `theta + mu` during evaluation.
2.  **PPO clipping:** Understand that PPO allows "safe" large updates. Variational noise can disrupt this "trust region" assumption if sigma is too large.
3.  **Slurm arrays:** Modifying the array index (`--array=0-29`) without understanding how it maps to Optuna trials can cause duplicate work or gaps in seeding.

---

# 21. Open Problems & Pending Refactors

1.  **Checkpointer:** Implement a `save_checkpoint(step)` and `load_checkpoint(path)` that includes `VecNormalize` stats.
2.  **Gradient Similarity:** The "Task Conflict" metric requires computing $\nabla_\theta L_{task_i}$ per task. This is currently too expensive ($O(N)$ backprops) to run every step. A "Periodic Diagnostic" mode is needed.

---

# 22. Resource & Constraint Context

*   **Experiment Budget:** 30 trials * 11 variants = 330 Jobs.
*   **Time Budget:** Each job is ~6 hours. Total compute = ~2000 GPU-hours.
*   **Constraint:** Must be completed before the "Conference Deadline" (Hypothetical, but treat as urgent).
