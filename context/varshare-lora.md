# VarShare + LoRA: Implementation Details
This document details the implementation of **VarShare-LoRA** (Variational Low-Rank Adaptation), a parameter-efficient extension of the standard VarShare architecture. Instead of learning a full-rank variational correction matrix ($\mu$) for every task, we decompose the correction into two low-rank stochastic matrices.
## 1. Core Concept
In standard `VarShare`, the effective weight for task $i$ is:
$$ W_i = \theta + \mu_i $$
where $\mu_i$ has the same dimensions as $\theta$ ($D_{out} \times D_{in}$). This scales linearly with the number of parameters in the network, which can be expensive for large backbones.
**VarShare-LoRA** approximates the variation $\mu_i$ using a low-rank decomposition:
$$ W_i = \theta + (B_i \cdot A_i) $$
where:
*   $A_i \in \mathbb{R}^{r \times D_{in}}$
*   $B_i \in \mathbb{R}^{D_{out} \times r}$
*   $r \ll \min(D_{in}, D_{out})$ is the rank (e.g., 4 or 8).
Crucially, in **VarShare**, $A_i$ and $B_i$ are **variational (stochastic)** parameters, not just deterministic weights.
## 2. Pytorch Implementation
The implementation is found in `models_improvements.py` as `VarShareLoRALayer`.
### A. Parameters
For a layer with input dimension $d_{in}$ and output dimension $d_{out}$:
1.  **Shared Backbone ($\theta$):**
    *   `self.theta`: Tensor of shape $(d_{out}, d_{in})$. Standard full-rank initialization.
2.  **Task-Specific Factors ($A, B$):**
    Instead of a full $\mu$ and $\rho$, we maintain posterior parameters for the factors:
    *   `self.mus_A`, `self.rhos_A`: Distributions for $A_i$.
    *   `self.mus_B`, `self.rhos_B`: Distributions for $B_i$.
### B. Initialization
*   **$\theta$**: Kaiming Uniform (standard).
*   **$A$ (Input Projection)**:
    *   $\mu_A \sim \mathcal{N}(0, 0.01)$: Small random initialization to break symmetry.
    *   $\rho_A = -5.0$: Leads to small initial variance.
*   **$B$ (Output Projection)**:
    *   $\mu_B = 0$: Initialized to zero. This ensures that at the start of training, $B \cdot A = 0$, so the effective weight is exactly the shared backbone $\theta$.
    *   $\rho_B = -5.0$: Small initial variance.
### C. Forward Pass (with Re-parameterization)
The forward pass involves sampling both matrices $A$ and $B$, or using the local reparameterization trick (though for matrix multiplication of two stochastic matrices, exact local reparameterization is complex, so we often simply sample weights in training for the residual).
```python
    def forward(self, x, task_id, temperature=1.0):
        # ... retrieve mu_A, rho_A, mu_B, rho_B ...
        
        # 1. Construct distributions
        sigma_A = F.softplus(rho_A) * temperature
        sigma_B = F.softplus(rho_B) * temperature
        
        # 2. Sample or Mean
        if self.training:
             A = Normal(mu_A, sigma_A).rsample()
             B = Normal(mu_B, sigma_B).rsample()
        else:
             A = mu_A
             B = mu_B
             
        # 3. Compute Effective Weight
        # W_eff = Theta + (B @ A)
        residual = B @ A
        weight = self.theta + residual
        
        return F.linear(x, weight, self.bias)
```
### D. KL Divergence Computation
Since $A$ and $B$ are independent in the variational posterior approximation, the total KL divergence is simply the sum of their individual KL divergences relative to the prior.
$$ KL_{total} = \sum KL(q(A_i) || p(A)) + \sum KL(q(B_i) || p(B)) $$
This drastically reduces the complexity penalty term compared to full-rank VarShare, potentially allowing for stronger regularization on the limited degrees of freedom.
## 3. Comparison to Baseline LoRA
| Feature | Standard LoRA | VarShare-LoRA |
| :--- | :--- | :--- |
| **Base Weight** | Fixed Pre-trained $W_0$ | Learned Shared $\theta$ |
| **Adaptation** | Deterministic $B \cdot A$ | Stochastic $B \cdot A$ |
| **Objective** | Maximize Reward | Maximize ELBO (Reward - KL) |
| **Use Case** | Fine-tuning LLMs | Bayesian Multi-Task RL |