# Theoretical Foundations of VarShare: Probabilistic Weights

## 1. Core Architecture
The original formulation of VarShare utilizes a Variational Bayesian approach, where task-specific residuals are injected with learned Gaussian noise during the forward pass:
$$ w_t = \theta + \mu_t + \sigma_t \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I) $$
- **$\sigma_t$**: The learned standard deviation (uncertainty) of the task-specific residual.
- **Objective**: The network optimizes both the RL objective and a Kullback-Leibler (KL) divergence penalty that forces the posterior distribution $\mathcal{N}(\mu_t, \sigma_t^2)$ towards a prior $\mathcal{N}(0, \sigma_{prior}^2)$.

---

## 2. Theoretical Advantages (The Promise)

### A. Targeted Parameter-Space Exploration
- **Theory**: Injecting noise directly into the weights (parameter space) rather than the actions (action space) generates temporally correlated, consistent exploratory behaviors over an entire episode.
- **Why it's "Targeted"**: Because $\sigma_t$ is learned, the network should theoretically act as an *auto-annealing exploration mechanism*. 
  - If a specific weight is crucial and uncertain, the KL penalty keeps $\sigma_t$ high, driving exploration.
  - Once the RL loss discovers the optimal value for that weight, the gradient from the RL loss saying "Stop varying this!" overpowers the KL penalty, forcing $\sigma_t \to 0$ and locking down the learned behavior.

### B. Finding Wide, Robust Minima
- **Theory**: Deterministic networks tend to sprint into the sharpest, deepest minima they can find in the loss landscape. These sharp minima are brittle; small shifts in the environment destroy performance.
- **Benefit**: Because $w_t$ is jittered by $\epsilon$ on every forward pass, the network *cannot* settle into a sharp ditch (the noise would instantly bounce it out, resulting in terrible loss). The optimizer is forced to find a massive, "wide valley" that is robust to constant parameter shaking. 
- **The Core Insight for Shared Backbones ($\theta$)**: Even though $\sigma_t$ only applies to the residual, the gradient for $\theta$ is evaluated *through* the noisy weight $w_t$. Therefore, the task-level noise acts as a structural smoothing mechanism, forcing the shared parameters to generalize exceptionally well and avoid brittle memorization.

---

## 3. Empirical Disadvantages & Failure Modes (The Reality)

### A. The Signal-to-Noise Problem (Collapse of $\sigma$)
- **The Flaw**: In practice, the targeted exploration fails completely. $\sigma_t$ immediately collapses to $\sigma_{prior}$ across all layers and stays there, becoming dumb, constant noise.
- **The Reason**: The gradient for the KL divergence ($\nabla_\sigma \mathcal{D}_{KL}$) has a perfect, smooth, analytical closed-form solution. Conversely, the RL gradient ($\nabla_\sigma \mathcal{L}_{RL}$) in methods like PPO is estimated via noisy trajectory rollouts. The optimizer treats the chaotic RL gradient as static noise and takes the path of least resistance: it perfectly satisfies the deterministic KL penalty by setting $\sigma_t = \sigma_{prior}$ immediately, destroying any intelligent adaptation.

### B. Inability to Converge on Sharp, Precise Subspaces (The "Trackmania" Problem)
- **The Flaw**: While wide minima are great for generalization, some tasks absolutely *require* sharp, hyper-precise parameter configurations to succeed (e.g., precise steering in a high-speed racing game, or a complex robotic manipulation task). 
- **The Reason**: Because $\sigma$ collapses to a constant $\sigma_{prior}$ instead of annealing to 0, the network never gets to stop exploring. It is like trying to thread a needle while someone is constantly vibrating your hand. The noise prevents the parameters from finally settling into the exact, narrow configuration needed for elite performance.

### C. Destruction of the PPO Surrogate Advantage
- **The Flaw**: This was empirically proven by our deterministic ablation (which matched Soft Modularization by simply turning $\sigma \to 0$).
- **The Reason**: PPO is an *on-policy* algorithm. It relies on taking multiple optimization epochs (e.g., 4 to 10) over the exact same batch of rollout data. Its math strictly assumes that the policy taking the update ($\pi_\theta$) is relatively close to the policy that collected the data ($\pi_{\theta_{old}}$). 
- By re-sampling $\epsilon$ during the PPO optimization epochs, the fundamental action distribution of the network changes radically underneath PPO's feet. PPO's clipped surrogate objective is violated, advantage estimation breaks down completely, and the value loss spikes.

---

## 4. Potential Solutions for Stochasticity

- **For Disadvantage A (KL Domination)**: Abandon the Bayesian KL penalty entirely. Use implementations like *NoisyNets*, where $\mu$ and $\sigma$ are learned exclusively via the RL loss. Alternatively, heavily anneal the KL penalty factor ($\beta$) starting from 0.0 to let the RL signal establish the variance landscape first.
- **For Disadvantage B (Uncontrolled Exploration)**: Introduce a global exponential decay on $\sigma_{prior}$ over the course of training, manually forcing the network from a highly stochastic exploratory regime into a deterministic exploitation regime (mimicking traditional $\epsilon$-greedy annealing).
- **For Disadvantage C (PPO Interference)**: 
  - Switch to an *off-policy* algorithm (like SAC or DDPG) which are significantly more robust to weight-space noise, as they do not rely on ratio clipping from a specific behavioral policy.
  - Mathematically enforce "Consistent Noise" sampling, where a single $\epsilon$ matrix is sampled at the beginning of an episode roll-out and kept perfectly frozen through all PPO update epochs (though early iterations of VarShare showed this alone is often not enough if the resulting policy is still too chaotic).
