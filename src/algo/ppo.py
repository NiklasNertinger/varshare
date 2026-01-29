
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    """
    PPO Algorithm implementation.
    """
    def __init__(self, agent, optimizer, 
                 lr=3e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_coef=0.2, 
                 ent_coef=0.0, 
                 vf_coef=0.5, 
                 max_grad_norm=0.5,
                 update_epochs=10, 
                 minibatch_size=64,
                 device='cpu',
                 kl_beta=0.0,
                 kl_schedule="fixed", # fixed, annealing, trigger
                 kl_controller_args={},
                 lambda_hyper=0.0): # For learned prior penalty
        
        self.agent = agent
        self.optimizer = optimizer
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = device
        
        # Variational Controller Args
        self.kl_beta = kl_beta
        self.base_kl_beta = kl_beta # Storage for annealing base
        self.kl_schedule = kl_schedule
        self.kl_controller_args = kl_controller_args
        self.lambda_hyper = lambda_hyper
        
        self.global_step = 0 # Track for annealing (approximated by updates)
        
        # Controller State
        self.ratio_ema = None 

    def update_kl_beta(self):
        """Updates self.kl_beta based on schedule/controller"""
        if self.kl_schedule == "fixed":
            return
            
        elif self.kl_schedule == "annealing":
            # Linear Warmup then Hold then Ramp
            total_updates = self.kl_controller_args.get("total_updates", 1000)
            warmup_frac = self.kl_controller_args.get("warmup_frac", 0.1)
            
            warmup_steps = total_updates * warmup_frac
            
            if self.global_step < warmup_steps:
                # Linear 0 -> base
                self.kl_beta = self.base_kl_beta * (self.global_step / max(1, warmup_steps))
            else:
                self.kl_beta = self.base_kl_beta
                
        elif self.kl_schedule == "trigger":
            # Adjust based on noise/weight ratio
            # This requires "current ratio" which is computed during loss or separate check
            # Implemented inside update loop after gathering metrics
            pass

    def get_noise_ratio(self):
        # Heuristic: Ratio of median sigma to median theta magnitude
        # We need to crawl the model
        sigmas = []
        thetas = []
        
        for name, mod in self.agent.named_modules():
            # Use specific VarShare/LoRA structure checks
            if hasattr(mod, "theta") and hasattr(mod, "rhos"):
                 thetas.append(mod.theta.detach().abs().median())
                 for k in mod.rhos:
                      sigmas.append(torch.nn.functional.softplus(mod.rhos[k].detach()).median())
            
            # Also check LoRA
            if hasattr(mod, "mus_A") and hasattr(mod, "rhos_A"):
                 thetas.append(mod.theta.detach().abs().median())
                 for k in mod.rhos_A:
                      sigmas.append(torch.nn.functional.softplus(mod.rhos_A[k].detach()).median())
                 for k in mod.rhos_B:
                      sigmas.append(torch.nn.functional.softplus(mod.rhos_B[k].detach()).median())

        if not sigmas or not thetas: return 0.0
        
        med_sigma = torch.stack(sigmas).median()
        med_theta = torch.stack(thetas).median() + 1e-6
        
        return (med_sigma / med_theta).item()

    def update(self, b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values, task_ids=None):
        """
        Perform PPO update on a batch of collected experience.
        Expects flattened batches (num_steps * num_envs, ...)
        
        task_ids: Optional tensor of task indices corresponding to the batch, for Multi-Task models.
        """
        # Calculate Advantages using GAE
        with torch.no_grad():
            # Note: This function assumes the buffer is already processed into GAE returns/advantages 
            # OR we process it here. Typically CleanRL processes GAE *before* flattening.
            # But here we receive flat tensors?
            # actually, for a generic PPO class, it might be better to receive the full "storage" object 
            # or pre-calculated advantages.
            # Let's assume we receive calculated advantages and returns to keep this class pure "Optimizer".
            pass
        raise NotImplementedError("Use update_from_storage instead for clarity.")

    def update_from_storage(self, obs, actions, logprobs, returns, advantages, values, task_ids=None):
        """
        obs: (N, ObsDim)
        actions: (N, ActionDim)
        logprobs: (N,)
        returns: (N,)
        advantages: (N,)
        values: (N,)
        task_ids: (N,) optional
        """
        self_global_step = self.global_step
        self.global_step += 1
        self.update_kl_beta()
        
        # Trigger Controller Logic (Pre-Update Check)
        if self.kl_schedule == "trigger" and (self_global_step % 20 == 0):
            current_ratio = self.get_noise_ratio()
            
            # EMA Update
            alpha = 0.05
            if self.ratio_ema is None: self.ratio_ema = current_ratio
            else: self.ratio_ema = (1-alpha)*self.ratio_ema + alpha*current_ratio
            
            # Band Logic
            target_ratio = self.kl_controller_args.get("target_ratio", 0.1)
            tol = 0.05 # +/- 5%
            
            if self.ratio_ema > (target_ratio + tol):
                self.kl_beta = min(self.kl_beta * 1.1, 100.0) # Increase penalty to reduce sigma
            elif self.ratio_ema < (target_ratio - tol):
                self.kl_beta = max(self.kl_beta / 1.1, 1e-6) # Reduce penalty to allow sigma

        b_inds = np.arange(len(obs))
        clipfracs = []
        approx_kl = 0 # fallback
        kl_loss = torch.tensor(0.0)
        grad_norm = 0.0

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(obs), self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                mb_task_ids = task_ids[mb_inds] if task_ids is not None else None
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    obs[mb_inds], 
                    actions[mb_inds], 
                    task_idx=mb_task_ids
                )
                
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                # Advantage normalization (optional, can be done at rollout-level instead)
                if getattr(self, 'norm_adv', True):
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                
                # Value Clipping (optional, CleanRL typically does it)
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                # --- VarShare Logic ---
                kl_loss = torch.tensor(0.0, device=self.device)
                
                if hasattr(self.agent, "get_kl") and mb_task_ids is not None:
                    # Expectation over tasks in batch: Average KL per task
                    unique_tasks = torch.unique(mb_task_ids)
                    batch_kl = 0
                    for t in unique_tasks:
                         batch_kl += self.agent.get_kl(t)
                    kl_val = batch_kl / len(unique_tasks)
                    
                    kl_loss = self.kl_beta * kl_val
                    loss += kl_loss

                # --- Empirical Bayes Hyperprior Penalty ---
                if self.lambda_hyper > 0:
                     target = math.log(0.1)
                     hyper_loss = torch.tensor(0.0, device=self.device)
                     for name, param in self.agent.named_parameters():
                          if "log_sigma_prior" in name:
                               hyper_loss += (param - target) ** 2
                     loss += self.lambda_hyper * hyper_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "kl_penalty": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "raw_kl": ((kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss) / self.kl_beta) if self.kl_beta > 0 else 0.0,
            "kl_beta": self.kl_beta,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        }
