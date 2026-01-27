
# VarShare RL: Variational Parameter Sharing for Multi-Task RL

Research framework for implementing and evaluating **VarShare**, a variational approach to parameter sharing in Deep Reinforcement Learning.

## Setup

1. **Environment**:
   The project uses a local virtual environment:
   ```bash
   .\.venv\Scripts\activate
   ```
   Dependencies are installed in `.venv`.

2. **Structure**:
   - `src/`: Core library code.
     - `algo/`: RL algorithms (Clean PPO implementation).
     - `env/`: Multi-task environment wrappers (`ComplexCartPole`, `IdenticalCartPole`).
     - `models.py`: Neural network architectures (Backbones + VarShare layers).
   - `scripts/`: Training and experimentation entrypoints.
   - `analysis/`: Automated output directory for logs and plots.

## Running Experiments

The primary training entrypoint is `scripts/train_varshare_ppo.py`. It uses fixed-horizon rollouts with GAE for maximum stability.

### Individual Training
```powershell
python scripts/train_varshare_ppo.py --env-type ComplexCartPole --total-timesteps 100000 --num-envs 8
```

### Comparison Study Runners
We have automated runners for specific experiment configurations:
- `scripts/run_exp_3_multitask_serial.py`: ComplexCartPole, 1 Env.
- `scripts/run_exp_4_multitask_parallel.py`: ComplexCartPole, 8 Envs.
- `scripts/run_exp_7_identical_serial.py`: IdenticalCartPole, 1 Env.
- `scripts/run_exp_8_identical_parallel.py`: IdenticalCartPole, 8 Envs.

## Logging & Analysis

- **Local Logging**: Every run saves a `heartbeat.csv` and automated plots (Reward, Norms, Sharing Ratio) to `analysis/<exp_name>/<seed_dir>/`.
- **WandB**: Integration is available but currently disabled in script for local stability.
- **Aggregation**: Use `python scripts/aggregate_plots.py` after multiple seeded runs to generate mean/std plots.

## Design Choices

- **Parallelization**: Fully supports vectorized environments via `SyncVectorEnv`.
- **Architecture**: Separate backbones for Actor and Critic to prevent gradient interference during parameter sharing.
- **VarShare**: Implements variational weights with learned means and log-variances, regularized by a KL penalty.
