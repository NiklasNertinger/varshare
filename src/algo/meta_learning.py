
import torch
import torch.nn as nn

def reptile_update(network, task_id, alpha=0.05):
    """
    Performs the Reptile meta-update logic on a given network.
    Shift shared parameters (theta) towards the task-specific solution,
    and adjust task-specific variation (mu) to preserve immediate behavior.
    
    Args:
        network (nn.Module): The network containing VarShareLayers.
        task_id (int): The ID of the task that was just trained.
        alpha (float): The step size for the meta-update. 
                       Higher values absorb task-specifics faster into shared params.
    """
    with torch.no_grad():
        task_key = str(task_id)
        count = 0
        
        # Iterate over all sub-modules (layers)
        for module in network.modules():
            # Check for VarShare duck-typing: has 'theta' and 'mus'
            if hasattr(module, 'theta') and hasattr(module, 'mus'):
                if task_key in module.mus:
                    # 1. Access the specific mu for the current task
                    mu = module.mus[task_key]
                    
                    # 2. Calculate the shift
                    shift = alpha * mu.data
                    
                    # 3. Apply the shift
                    # Move shared theta towards the task solution
                    # theta_new = theta_old + alpha * mu
                    module.theta.data.add_(shift)
                    
                    # Adjust mu to preserve effective weight w = theta + mu
                    # mu_new = mu_old - alpha * mu
                    # Then theta_new + mu_new = theta_old + alpha*mu + mu_old - alpha*mu = theta_old + mu_old (preserved)
                    mu.data.sub_(shift)
                    count += 1
                    
    return count
