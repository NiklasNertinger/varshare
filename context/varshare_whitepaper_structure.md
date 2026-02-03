# VarShare Methodology Whitepaper - Proposed Structure

## 1. Introduction
- **The Multi-Task Learning (MTL) Dilemma**: Balancing positive transfer (sharing knowledge) with avoiding negative transfer (task conflict).
- **Existing Approaches**: 
    - Hard Parameter Sharing (Shared Backbone).
    - Soft Parameter Sharing (Constraints).
    - Modularization (Routing/MoE).
- **The VarShare Proposition**: A probabilistic approach where task-specific parameters are modeled as variational deviations from a shared global "prior".

## 2. Preliminaries
- **Reinforcement Learning Setting**: MDPs, PPO algorithm basics.
- **Bayesian Neural Networks (BNN)**: Brief primer on weights as distributions.

## 3. Methodology: Variational Parameter Sharing (VarShare)
### 3.1. Probabilistic Formulation
- Define the weight for task $t$ as a random variable $w^{(t)}$.
- **The Shared Prior**: The global parameters $\theta$ serve as the mean of the prior distribution for all tasks.
  $$ p(w^{(t)} | \theta) = \mathcal{N}(\theta, \sigma_{prior}^2 I) $$
- **The Variational Posterior**: We approximate the intractable posterior with a factorized Gaussian:
  $$ q(w^{(t)}) = \mathcal{N}(\theta + \mu^{(t)}, \text{diag}(\sigma^{(t)^2})) $$
  where $\mu^{(t)}$ is the task-specific shift and $\sigma^{(t)}$ is the task-specific uncertainty.

### 3.2. The Reparameterization Trick
- How to backpropagate through sampling.
- Forward pass equation:
  $$ w^{(t)}_{effective} = \theta + \mu^{(t)} + \text{softplus}(\rho^{(t)}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
- Explanation of the role of $\epsilon$ (noise injection for exploration/regularization).

### 3.3. Optimization Objective (The Loss Function)
- Derivation of the Evidence Lower Bound (ELBO) in the RL context.
- **The Reconstruction Term**: $\mathbb{E}_{q}[\mathcal{L}_{RL}]$ (Standard PPO Loss using sampled weights).
- **The Regularization Term**: Kullback-Leibler (KL) Divergence between Posterior and Prior.
  - Derivation of the analytical KL for Gaussians.
  - Interpretation: The "elastic force" pulling tasks back to the shared mean $\theta$.
  - The $\beta_{KL}$ hyperparameter (Lagrange multiplier for the constraint).

## 4. Scalable Variants
### 4.1. VarShare-LoRA (Low-Rank Adaptation)
- **Problem**: Standard VarShare triples parameter count per task ($2N$ parameters for $\mu, \sigma$). Scales poorly with Large Models (LLMs/Transformers) or many tasks.
- **Solution**: Apply VarShare only to low-rank adapters.
  $$ W^{(t)} = \theta + B^{(t)}A^{(t)} $$
  where $A^{(t)}, B^{(t)}$ are variational.
- Complexity analysis: $O(d \times k)$ vs $O(d^2)$.

### 4.2. Empirical Bayes (Learned Prior)
- **Problem**: Tuning fixed $\sigma_{prior}$ is difficult.
- **Solution**: Treat the prior variance $\sigma_{prior}$ as a learnable parameter ($\Sigma$).
- The "Hyperprior" penalty (optional).

### 4.3. Partial Sharing Architectures
- Hypothesis: Lower layers learn general features (edges, physics), deeper layers specialize.
- Application: Standard shared layers for $L_{1}..L_{k}$, VarShare layers for $L_{k+1}..L_{out}$.

## 5. Implementation Details
- **Architecture**:
    - Backbone vs Heads.
    - Handling Task Embeddings (or lack thereof).
- **PPO Integration**:
    - Handling value bootstrapping with stochastic weights.
    - Annealing schedules for $\beta_{KL}$ to prevent premature collapse.

## 6. Experimental Setup
- **Environment**: Meta-World MT4 (Custom subset: Window-Close, Push, Pick-Place, Door-Open).
- **Baselines**:
    - Independent PPO (No sharing).
    - Shared PPO (Hard sharing).
    - PaCo (Parameter Composition).
    - Soft Modularization.
- **Metrics**: Success Rate, Sample Efficiency, Architectural Analysis (Norm of $\mu$ vs $\theta$).

## 7. Current Challenges & Discussion
- **Posterior Collapse**: Conditions under which $\mu^{(t)} \to 0$ and the model degenerates to Hard Sharing.
- **Noise Analysis**: Does the noise $\sigma^{(t)}$ aid exploration or just hinder convergence?
- **Future Directions**: 
    - Adaptive $\beta_{KL}$ (Trigger-based).
    - Application to Transformers/LLMs.
