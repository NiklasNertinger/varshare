import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from scipy.optimize import minimize

class CAGrad:
    def __init__(self, optimizer, c=0.4):
        """
        Conflict-Averse Gradient Descent Engine
        c: Hyperparameter controlling the strictness of the neighborhood projection. 
           Standard values range from 0.2 to 0.7.
        """
        self.optimizer = optimizer
        self.c = c

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
        
        # 2. Project conflicting gradients via CAGrad
        final_grad = self._project_grads(task_grads)
        
        # 3. Apply final gradient to parameters
        self._set_grad(final_grad)

    def _project_grads(self, task_grads):
        """
        task_grads: list of flattened task gradients
        """
        grads = torch.stack(task_grads) # (num_tasks, dim)
        num_tasks = grads.size(0)
        
        if num_tasks <= 1:
            return grads.sum(dim=0)
            
        g0 = grads.mean(dim=0)
        
        # Inner product matrix GG
        GG = torch.mm(grads, grads.t()).cpu().numpy() # [K, K]
        g0_norm = (g0.pow(2).sum() + 1e-8).sqrt().item()
        
        # Dual Objective for finding weighting vector w
        def obj(w):
            gw_norm_sq = w.T.dot(GG).dot(w) + 1e-8
            return w.T.dot(GG.sum(1) / num_tasks) + self.c * g0_norm * np.sqrt(gw_norm_sq)
            
        w0 = np.ones(num_tasks) / num_tasks
        bnds = tuple((0.0, 1.0) for _ in w0)
        cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0})
        
        # Solve dual using SciPy SLSQP
        res = minimize(obj, w0, bounds=bnds, constraints=cons)
        w = res.x
        
        # Reconstruct descent direction d from optimal weights w
        gw = (grads * torch.tensor(w, dtype=torch.float32, device=grads.device).view(-1, 1)).sum(dim=0)
        gw_norm = (gw.pow(2).sum() + 1e-8).sqrt()
        
        d = g0 + self.c * g0_norm * gw / gw_norm
        return d

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
