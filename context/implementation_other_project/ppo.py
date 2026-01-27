import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, policy_net, value_net, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, kl_beta=0.01, l2_lambda=0.01, entropy_coef=0.01, device='cpu', action_std_init=None):
        self.policy = policy_net
        self.value = value_net
        self.optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.mse_loss = nn.MSELoss()
        self.kl_beta = kl_beta  # For VarShare
        self.l2_lambda = l2_lambda # For SoftSharing
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Continuous Action Support
        self.action_std = action_std_init
        if self.action_std is not None:
             self.action_var = torch.full((policy_net.output_layer.out_features,), action_std_init * action_std_init).to(device) # Assuming 1D action space per env if vector
        else:
             self.action_var = None

    def update(self, memory, task_id, is_varshare=False, **forward_kwargs):
        # memory is a list of (state, action, log_prob, reward, done)
        states = torch.FloatTensor(np.array([t[0] for t in memory])).to(self.device)
        
        if self.action_std is not None:
            actions = torch.FloatTensor(np.array([t[1] for t in memory])).to(self.device)
        else:
            actions = torch.FloatTensor(np.array([t[1] for t in memory])).to(self.device)
            
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in memory])).to(self.device)
        rewards = [t[3] for t in memory]
        dones = [t[4] for t in memory]
        
        # Calculate Returns / Advantages
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Optimization steps
        for _ in range(self.k_epochs):
            self.policy.train()
            self.value.train()
            
            # Forward Pass - Action Distribution
            action_output = self.policy(states, task_id, **forward_kwargs)
            
            if self.action_std is not None:
                # Continuous: Output is Mean
                # Fixed Std Dev for now (simplest PPO variant for fine-tuning)
                # Or we can anneal it. For now, use the fixed var initialized.
                cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
                dist = torch.distributions.MultivariateNormal(action_output, cov_mat)
            else:
                # Discrete: Output is Logits
                dist = torch.distributions.Categorical(logits=action_output)
            
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()
            
            state_values = self.value(states, task_id).squeeze()
            
            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.mse_loss(state_values, returns)
            
            loss = loss_actor + 0.5 * loss_critic - self.entropy_coef * dist_entropy
            
            if is_varshare:
                kl_penalty = self.policy.get_kl(task_id) + self.value.get_kl(task_id)
                loss += self.kl_beta * (kl_penalty / len(states))
            
            if hasattr(self.policy, 'get_l2_penalty') and self.l2_lambda > 0:
                 l2_pen = self.policy.get_l2_penalty(task_id) + self.value.get_l2_penalty(task_id)
                 loss += self.l2_lambda * l2_pen
            
            self.optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in list(self.policy.parameters()) + list(self.value.parameters()):
                 if p.grad is not None:
                     total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            self.optimizer.step()
            
            # Record metrics
            metrics_epoch = {
                'loss': loss.item(),
                'loss_actor': loss_actor.item(),
                'loss_critic': loss_critic.item(),
                'entropy': dist_entropy.item(),
                'grad_norm': total_norm
            }
            if is_varshare:
                 metrics_epoch['kl_penalty'] = (kl_penalty.item() / len(states))
                 metrics_epoch['kl_raw'] = kl_penalty.item()
            
            metrics_epoch['grad_steps'] = self.k_epochs
            
            # For simplicity, just return the last epoch's metrics, or average?
            # Averaging over k_epochs is safer for logging.
            
        # Return last metrics for simplicity (or we can average if we want to be fancy)
        return metrics_epoch

    def update_multitask(self, memories, algo='independent'):
        # memories: dict {task_id: memory_list} or list of [memory_task_0, memory_task_1...]
        # We assume memories is a list/dict keyed by task_idx.
        
        # 1. Flatten/Process Data per Task
        task_data = {}
        for task_id, mem in enumerate(memories):
            if not mem: continue # Skip if empty
            states = torch.FloatTensor(np.array([t[0] for t in mem])).to(self.device)
            actions = torch.FloatTensor(np.array([t[1] for t in mem])).to(self.device)
            old_log_probs = torch.FloatTensor(np.array([t[2] for t in mem])).to(self.device)
            rewards = [t[3] for t in mem]
            dones = [t[4] for t in mem]
            
            # Returns
            returns = []
            discounted_sum = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done: discounted_sum = 0
                discounted_sum = reward + (self.gamma * discounted_sum)
                returns.insert(0, discounted_sum)
            returns = torch.FloatTensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            
            task_data[task_id] = (states, actions, old_log_probs, returns)

        # 2. Optimization Loop
        # For PPO, we do K epochs. In MT setting, we usually shuffle tasks or do rigorous updates.
        # PCGrad requires gradients for ALL tasks at current point.
        
        avg_loss = 0
        
        for _ in range(self.k_epochs):
            task_grads = []
            task_losses = []
            
            self.optimizer.zero_grad() # Clear global
            
            # --- Step A: Compute Gradients per Task Indepedently ---
            for task_id, (states, actions, old_log_probs, returns) in task_data.items():
                self.policy.train()
                self.value.train()
                
                # Forward
                action_logits = self.policy(states, task_id)
                dist = torch.distributions.Categorical(logits=action_logits)
                log_probs = dist.log_prob(actions)
                dist_entropy = dist.entropy().mean()
                state_values = self.value(states, task_id).squeeze()
                
                ratios = torch.exp(log_probs - old_log_probs.detach())
                advantages = returns - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.mse_loss(state_values, returns)
                
                loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy
                
                # Backward to get gradients for THIS task ONLY
                # We need to be careful not to accumulate yet if using PCGrad.
                # However, PyTorch accumulates into .grad.
                # So we must clear, backward, copy, clear.
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Save grads
                grads = []
                for p in self.policy.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.clone())
                    else:
                        grads.append(torch.zeros_like(p))
                # Value net params too? Yes, usually shared backbone.
                # But self.optimizer has both.
                # Let's iterate optimizer param_groups to be safe, or just list(policy)+list(value).
                # Note: if policy/value share weights, we must be careful.
                # In models.py they are distinct objects usually, unless shared backbone is object reference?
                # In SharedNetwork, they are distinct instances.
                # In Contextual, distinct instances.
                # Wait, if they are separate networks, PCGrad between Actor Task A and Actor Task B makes sense.
                # But Actor Task A and Critic Task A?
                # Usually we sum Actor+Critic loss per task.
                
                # Collecting ALL params in optimizer
                all_params = []
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        all_params.append(p)
                
                # Re-reading grads from all_params
                current_task_grads = []
                for p in all_params:
                     if p.grad is not None:
                         current_task_grads.append(p.grad.clone())
                     else:
                         current_task_grads.append(torch.zeros_like(p))
                
                task_grads.append(current_task_grads)
                task_losses.append(loss.item())

            # --- Step B: Apply PCGrad / CAGrad ---
            
            final_grads = None
            
            if algo == 'pcgrad':
                final_grads = self._pcgrad_project(task_grads)
            elif algo == 'cagrad':
                 final_grads = self._cagrad_solve(task_grads)
            else:
                # Default: Just sum/average them (Standard MTL)
                final_grads = self._average_grads(task_grads)
            
            # --- Step C: Apply to Model ---
            self.optimizer.zero_grad()
            
            # Log Grad Norm
            total_norm = 0.0
            idx = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    g = final_grads[idx]
                    if g is not None:
                        total_norm += g.norm().item()
                    p.grad = g # Set the gradient
                    idx += 1
            
            self.optimizer.step()
            avg_loss += np.mean(task_losses)

        return avg_loss / self.k_epochs, total_norm, self.k_epochs * len(task_data)

    def _average_grads(self, task_grads):
        # Simply average gradients across tasks
        if not task_grads: return []
        num_tasks = len(task_grads)
        num_params = len(task_grads[0])
        
        final_grads = []
        for p_idx in range(num_params):
            sum_g = task_grads[0][p_idx].clone() * 0 # Init to zero
            count = 0
            for t_idx in range(num_tasks):
                # Ensure we handle None grads if some params are task-specific (e.g. heads)
                # But task_grads[t_idx] list should preserve order.
                # If a param computed grad, it's there. Else zero?
                # We initialized zeros in the loop above.
                sum_g += task_grads[t_idx][p_idx]
                count += 1
            final_grads.append(sum_g / num_tasks)
        return final_grads

    def _pcgrad_project(self, task_grads):
        import random
        num_tasks = len(task_grads)
        if num_tasks == 0: return []

        proj_grads = [[g.clone() for g in t_grads] for t_grads in task_grads]
        
        indices = list(range(num_tasks))
        random.shuffle(indices) 
        
        for i in indices:
            random_others = list(range(num_tasks))
            random.shuffle(random_others)
            
            for j in random_others:
                if i == j: continue
                
                # Compute dot product
                g_i_flat = torch.cat([g.flatten() for g in proj_grads[i]])
                g_j_flat = torch.cat([g.flatten() for g in proj_grads[j]])
                
                dot_prod = torch.dot(g_i_flat, g_j_flat)
                
                if dot_prod < 0:
                    denom = torch.dot(g_j_flat, g_j_flat) + 1e-8
                    scale = dot_prod / denom
                    
                    for p_idx in range(len(proj_grads[i])):
                        proj_grads[i][p_idx] -= scale * proj_grads[j][p_idx]
                        
        # Sum projected
        final = []
        num_params = len(proj_grads[0])
        for p_idx in range(num_params):
             sum_g = proj_grads[0][p_idx].clone()
             for t_idx in range(1, num_tasks):
                 sum_g += proj_grads[t_idx][p_idx]
             final.append(sum_g) 
             
        return final

    def _cagrad_solve(self, task_grads):
        """
        Solves the MGDA / Min-Norm problem:
        min_w || sum_i w_i g_i ||^2 s.t. sum w_i = 1, w_i >= 0
        """
        from scipy.optimize import minimize
        num_tasks = len(task_grads)
        num_params = len(task_grads[0])
        
        # Flatten grads per task for optimization
        # shape: (num_tasks, num_total_params)
        grads_flat = []
        for t in range(num_tasks):
            grads_flat.append(torch.cat([g.flatten() for g in task_grads[t]]).cpu().numpy())
        
        grads_matrix = np.array(grads_flat) # (N, D)
        # Goal: Find w (N,) to min || w^T G ||^2 = w^T G G^T w
        # Let M = G G^T (NxN matrix of dot products)
        
        M = grads_matrix @ grads_matrix.T # (N, N)
        
        # Objective:0.5 * w^T M w
        def obj(w):
            return 0.5 * (w @ M @ w)
        
        def jac(w):
            return M @ w
        
        # Constraints: sum w = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        # Bounds: w >= 0
        bounds = [(0, 1) for _ in range(num_tasks)]
        
        w0 = np.ones(num_tasks) / num_tasks
        
        res = minimize(obj, w0, method='SLSQP', jac=jac, bounds=bounds, constraints=constraints)
        
        weights = res.x
        
        # Compute final gradient: sum w_i g_i
        # But we want to maximize improvement... MGDA gives descent direction.
        # usually we multiply by N or sum to keep scale similar to standard gradient descent
        # Min-Norm usually strictly shrinks gradient.
        # CAGrad paper suggests rescaling or using this direction.
        # We will use the weighted sum direction, scaled by N to match magnitude of "sum loss".
        
        final_grads = []
        for p_idx in range(num_params):
            weighted_g = task_grads[0][p_idx].clone() * weights[0]
            for t_idx in range(1, num_tasks):
                weighted_g += task_grads[t_idx][p_idx] * weights[t_idx]
            
            # Rescale? If we use sum in standard, then avg = 1/N sum.
            # Here w sums to 1. So it's like average.
            # If standard update uses sum of gradients, it's roughly N times bigger.
            # We should probably scale by num_tasks to keep LR roughly consistent.
            final_grads.append(weighted_g * num_tasks)
            
        return final_grads
