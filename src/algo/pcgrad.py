import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random

class PCGrad:
    def __init__(self, optimizer):
        self.optimizer = optimizer

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
        pc_grads = copy.deepcopy(task_grads)
        
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
                    g_i -= (dot_product / (g_j.norm()**2 + 1e-8)) * g_j
            
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
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.data.flatten())
        return torch.cat(grads)

    def _set_grad(self, flattened_grad):
        """
        Sets the flattened gradient back to the model parameters.
        """
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.grad = flattened_grad[idx:idx + numel].view(p.shape).clone()
                idx += numel
