# VarShare + Reptile: Meta-Learning Implementation Details
This document details the specific implementation of the **Reptile Meta-Update** within our `VarShare` architecture. This approach allows the shared parameters ($\theta$) to slowly drift towards a centroid that is beneficial for all tasks, effectively "absorbing" common task structures into the shared backbone.
## 1. Core Concept
In standard `VarShare`, effective weights are composed of a shared parameter and a task-specific variation:
$$ w_i = \theta + \mu_i $$
The **Reptile update** moves the shared initialization $\theta$ towards the current task's optimal solution ($w_i$), while simultaneously adjusting $\mu_i$ to keep the *current* effective weights unchanged. This ensures that the immediate performance on the current task is not disrupted, but the initialization point for *other* tasks (and future learning) improves.
**Update Rule:**
1.  **Shift Calculation:** $\Delta = \alpha \cdot \mu_i$
2.  **Shared Update:** $\theta \leftarrow \theta + \Delta$
3.  **Task Update:** $\mu_i \leftarrow \mu_i - \Delta$
**Result:**
$$ (\theta + \Delta) + (\mu_i - \Delta) = \theta + \mu_i $$
The effective weight $w_i$ remains mathematically identical, but the "load" is shifted from the specific parameter $\mu_i$ to the shared parameter $\theta$.
## 2. Pytorch Implementation
The implementation is housed within the `SAC` class in `sac.py`. It is designed to be agnostic to the specific network depth, relying on inspecting module attributes.
### A. The `meta_update` Method
This method is called periodically (e.g., at the end of every episode or after $N$ gradient steps) for the active `task_id`.
```python
def meta_update(self, task_id, alpha=0.05):
    """
    Performs the Reptile meta-update logic on Policy, Q1, and Q2 networks.
    
    Args:
        task_id (int): The ID of the task that was just trained.
        alpha (float): The step size for the meta-update (default: 0.05).
                       Higher values absorb task-specifics faster into shared params.
    """
    with torch.no_grad():
        # Iterate over all networks involved in the RL algorithm
        for net in [self.policy, self.q1, self.q2]:
            
            # Iterate over all sub-modules (layers)
            for module in net.modules():
                
                # Check for VarShare duck-typing
                # We look for 'theta' (shared) and 'mus' (task-specific dict)
                if hasattr(module, 'theta') and hasattr(module, 'mus'):
                    
                    # 1. Access the specific mu for the current task
                    # Note: 'module.mus' is a nn.ParameterDict
                    mu = module.mus[str(task_id)]
                    
                    # 2. Calculate the shift
                    shift = alpha * mu.data
                    
                    # 3. Apply the shift
                    # Move shared theta towards the task solution
                    module.theta.data.add_(shift)
                    
                    # Adjust mu to preserve effective weight w = theta + mu
                    module.mus[str(task_id)].data.sub_(shift)
    # 4. Sync Target Networks
    # Since Q1/Q2 shared parameters have changed, we must re-sync targets.
    # We choose to do a hard update (copy) here to ensure consistency.
    self.q1_target = copy.deepcopy(self.q1).to(self.device)
    self.q2_target = copy.deepcopy(self.q2).to(self.device)
```
### B. Key Requirements in `models.py`
For the above updater to work, your `VarShareLayer` must strictly expose:
1.  `self.theta`: A `nn.Parameter` representing shared weights.
2.  `self.mus`: A `nn.ParameterDict` mapping `str(task_id)` to `nn.Parameter`.
**Example `VarShareLayer` structure:**
```python
class VarShareLayer(nn.Module):
    def __init__(self, in_features, out_features, ...):
        super().__init__()
        # Shared Parameter
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Task Variations
        self.mus = nn.ParameterDict() 
        # ... logic to populate self.mus[str(task_id)] ...
```
## 3. Integration Logic
When integrating this into the training loop (`train.py` or `quick_experiment.py`):
1.  **Standard Update**: Perform standard SAC gradient updates (Adam) on $\theta$ and $\mu_i$ to minimize the expected return loss.
2.  **Meta Trigger**: At the end of a specific interval (e.g., `done` is True), trigger the meta-update.
3.  **Concurrency**: If running multiple tasks, ensure `meta_update` handles race conditions if updates are parallel (standard in our serial loop, this is safe).
```python
# Pseudo-code usage in Training Loop
agent.update(replay_buffer, task_id)
if done:
    # "Absorb" 5% of the task-specific deviation into the shared backbone
    agent.meta_update(task_id, alpha=0.05)
```
