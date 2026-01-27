
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
                 device='cpu'):
        
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
        
        b_inds = np.arange(len(obs))
        clipfracs = []
        
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
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        }
