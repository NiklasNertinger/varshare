# VarShare Methodology Summary (v2)

This document summarizes the current state of the **VarShare (Variational Parameter Sharing)** project for Multi-Task Reinforcement Learning.

## 1. Core Concept: VarShare
**VarShare** is a method for efficient multi-task learning that balances **sharing** knowledge across tasks with **specializing** for individual tasks.

### Mathematical Formulation
For a neural network weight parameters $w^{(t)}$ for task $t$:
$$ w^{(t)} = \theta + \delta^{(t)} $$
*   $\theta$: **Shared parameters** (global knowledge).
*   $\delta^{(t)}$: **Task-specific deviation**.

We treat the task-specific parameters as random variables with a variational posterior:
$$ q(\delta^{(t)} | \theta) = \mathcal{N}(\mu^{(t)}, \text{diag}(\sigma^{(t)})) $$
where $\sigma^{(t)} = \text{softplus}(\rho^{(t)})$.

During training (Forward Pass):
1.  Sample noise $\epsilon \sim \mathcal{N}(0, I)$.
2.  Compute effective weight: $w^{(t)} = \theta + \mu^{(t)} + \sigma^{(t)} \odot \epsilon$ (using the reparameterization trick).
3.  Compute output: $y = x w^{(t)^T} + b$.

### Loss Function
The objective is to maximize the expected return while keeping task parameters close to the shared "prior" (regularization towards the mean).
$$ \mathcal{L} = \mathcal{L}_{PPO} + \beta_{KL} \sum_{t} D_{KL}(q(\delta^{(t)}) || p(\delta^{(t)})) $$
*   Prior $p(\delta^{(t)}) = \mathcal{N}(0, \sigma^2_{prior})$.
*   $\beta_{KL}$: Hyperparameter controlling the strength of the "pull" towards the shared parameters.

## 2. Variations Implemented

### A. Standard VarShare (`mt10_varshare_base`)
*   Fixed Prior: $\sigma_{prior}$ is a fixed hyperparameter (default 1.0 or searched).
*   Architecture: All linear layers are `VarShareLayer`.

### B. Empirical Bayes / Learned Prior (`mt10_varshare_emp_bayes`)
*   Instead of fixing $\sigma_{prior}$, we **learn** it as a parameter $\Sigma$.
*   This allows the network to decide how "rigid" the sharing should be.

### C. LoRA VarShare (`mt10_varshare_lora`)
*   Instead of variational parameters on the full weight matrix, we use Low-Rank Adaptation.
*   $W^{(t)} = \theta + B^{(t)} A^{(t)}$
*   $A^{(t)}, B^{(t)}$ are task-specific variational distributions (Low Rank).
*   Reduces parameter count significantly.

### D. Partial Sharing (`mt10_varshare_partial`)
*   Standard (independent) layers for the initial layers.
*   `VarShareLayer` only for the last hidden layer and the output head.
*   Hypothesis: Early layers learn general features that don't need specialization.

### E. Reptile Initialization (`mt10_varshare_reptile`)
*   Initializes $\theta$ using Meta-Learning (Reptile algorithm) before starting VarShare training.

## 3. Environment Setting: "MT4"
We are currently validating on a custom **4-Task Benchmark** derived from Meta-World "MT10".
*   **Tasks:**
    1.  `window-close-v3`: Hard, articulation.
    2.  `push-v3`: Medium, object manipulation.
    3.  `pick-place-v3`: Hard, precision.
    4.  `door-open-v3`: Medium, articulation.
*   **Scale:** 35 Trials per study.
*   **Horizon:** 3,000,000 environment steps.
*   **Baselines:** Shared PPO (Upper bound?), Independent PPO, PCGrad, PaCo, Soft Modularization.

## 4. Current Challenges & Focus
*   **Posterior Collapse:** The model might ignore the latent task variables ($\mu, \sigma$) and just learn everything in $\theta$ if $\beta_{KL}$ is too high.
*   **Constraint vs. Freedom:** Finding the right balance (`kl_beta`, `prior_scale`) so tasks can specialize without drifting too far.
*   **Scalability:** VarShare adds $2 \times N_{params}$ per task (Mu, Rho). LoRA is the solution here.

## 5. Key Files
*   `src/models.py`: Definitions of `VarShareLayer` and `VarShareLoRALayer`.
*   `src/algo/ppo.py`: PPO algorithm adapted to handle the KL penalty.
*   `scripts/train_varshare_ppo.py`: Main training loop.
*   `src/env/metaworld_wrapper.py`: Environment definition (MT4).
