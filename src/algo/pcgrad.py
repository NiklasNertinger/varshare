import torch
import torch.nn as nn
import torch.optim as optim
import random

class PCGrad:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._zero_grad_cache = {} # Cache for zero tensors to prevent allocation overhead

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def pc_backward(self, objectives):
        """
        objectives: list of scalar losses, one per task
        """
        # 1. Compute per-task gradients
        task_grads = []
        for obj in objectives:
            self.optimizer.zero_grad()
            obj.backward(retain_graph=True)
            task_grads.append(self._get_grad())
        
        # 2. Project conflicting gradients
        final_grad = self._project_grads(task_grads)
        
        # 3. Apply final gradient to parameters
        self._set_grad(final_grad)

    def _project_grads(self, task_grads):
        """
        task_grads: list of flattened task gradients
        """
        num_tasks = len(task_grads)
        # Use .clone() instead of the extremely slow copy.deepcopy()
        pc_grads = [g.clone() for g in task_grads]
        
        # Shuffle tasks to avoid bias
        indices = list(range(num_tasks))
        random.shuffle(indices)
        
        for i in indices:
            # i-th task gradient
            g_i = pc_grads[i]
            
            # Others
            other_indices = [j for j in indices if j != i]
            random.shuffle(other_indices)
            
            for j in other_indices:
                g_j = task_grads[j]
                
                # Compute inner product
                dot_product = torch.dot(g_i, g_j)
                if dot_product < 0:
                    # Resolve conflict: project g_i onto normal plane of g_j
                    norm_j_sq = torch.dot(g_j, g_j) + 1e-8
                    g_i = g_i - (dot_product / norm_j_sq) * g_j
            
            pc_grads[i] = g_i
            
        # Sum of projected gradients
        return torch.stack(pc_grads).sum(dim=0)

    def _get_grad(self):
        """
        Retrieves flattened gradient from the model parameters handled by the optimizer.
        """
        grads = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    # Use cached zero tensors to prevent massive memory allocation overhead 
                    # for task-specific heads not active in the current backward pass.
                    if p not in self._zero_grad_cache:
                        self._zero_grad_cache[p] = torch.zeros(p.numel(), device=p.device)
                    grads.append(self._zero_grad_cache[p])
                else:
                    grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def _set_grad(self, flattened_grad):
        """
        Sets the flattened gradient back to the model parameters.
        """
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.grad = flattened_grad[idx:idx + numel].view_as(p).clone()
                idx += numel
