import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import collections
import csv
import json
import matplotlib.pyplot as plt
import wandb
import signal

shutdown_requested = False
def sigusr1_handler(signum, frame):
    global shutdown_requested
    print("\n[WARNING] Caught SIGUSR1 from SLURM! Will checkpoint and exit after current update.", flush=True)
    shutdown_requested = True

if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, sigusr1_handler)

from src.env import ComplexCartPole, IdenticalCartPole, MetaWorldWrapper
from src.env.toy_multitask import MultiTaskLunarLander, IdenticalLunarLander
from src.models_routing import DetActorCritic
from src.algo.ppo import PPO

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="det_varshare_base", help="experiment name")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="total timesteps")
    parser.add_argument("--n-steps", type=int, default=256, help="rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch size")
    parser.add_argument("--num-envs", type=int, default=1, help="number of parallel envs")
    
    # Golden Hyperparameters
    parser.add_argument("--lr-actor", type=float, default=0.001, help="actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=0.0001, help="critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="gae lambda")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="clip coefficient")
    parser.add_argument("--k-epochs", type=int, default=4, help="update epochs")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Max gradient norm for clipping")
    
    # DetVarShare specific
    parser.add_argument("--mu-l2-coef", type=float, default=0.001, help="L2 penalty coefficient for mu weights")
    parser.add_argument("--mu-init", type=float, default=0.0, help="Mu init")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Backbone hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--variant", type=str, default="base", 
        choices=["base", "lora", "gated", "l1", "film", "pcgrad", "cagrad", "decay", "hyperprior", "ara", "routing"], 
        help="Deterministic Architectural Variant")
    parser.add_argument("--lora-rank", type=int, default=4, help="Rank for LoRA (only used if variant=lora)")
    parser.add_argument("--cagrad-c", type=float, default=0.4, help="CAGrad neighborhood constant c")
    parser.add_argument("--routing-alpha", type=float, default=0.1, help="Routing strength")
    parser.add_argument("--routing-lambda", type=float, default=0.001, help="Routing adapter shrinkage")
    
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="cuda toggle")
    parser.add_argument("--compile", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use torch.compile to optimize PyTorch execution")
    
    parser.add_argument("--eval-mode", type=bool, default=True, help="Run periodic evaluations")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Eval every N steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per task during eval")
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Root analysis directory")
    
    # Environment selection
    parser.add_argument("--env-type", type=str, default="ComplexCartPole", 
        choices=["ComplexCartPole", "IdenticalCartPole", "metaworld", "MultiTaskLunarLander", "IdenticalLunarLander"], 
        help="Environment type")
    parser.add_argument("--mt-setting", type=str, default="MT10", choices=["MT1", "MT3", "MT4", "MT10", "MT50", "None"], help="Meta-World setting")
    parser.add_argument("--wandb-project", type=str, default="varshare-bench-mt10", help="W&B project name")
    
    args = parser.parse_args()
    return args

def calculate_explained_variance(y_true, y_pred):
    var_y = np.var(y_true)
    if var_y == 0: return 0
    return 1 - np.var(y_true - y_pred) / var_y

def evaluate(agent, env, device, num_episodes=5, num_tasks=5):
    agent.eval()
    task_rewards = {}
    for t_idx in range(num_tasks):
        rewards = []
        successes = []
        for _ in range(num_episodes):
            env.reset_task(t_idx)
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_success = 0
            while not done:
                obs_t = torch.Tensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _, _ = agent.get_action_and_value(obs_t, task_idx=t_idx, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                ep_reward += reward
                if "success" in info:
                    ep_success = max(ep_success, info["success"])
            rewards.append(ep_reward)
            successes.append(ep_success)
        task_rewards[t_idx] = {"reward": np.mean(rewards), "success": np.mean(successes)}
    agent.train()
    return task_rewards

def train(report_callback=None):
    args = parse_args()
    timestamp = int(time.time())
    run_name = f"{args.exp_name}__seed{args.seed}__{timestamp}"
    
    exp_dir = os.path.join(args.analysis_dir, args.exp_name)
    seed_dir = os.path.join(exp_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # W&B Initialization
    use_wandb = False
    try:
        wandb.init(
            project=args.wandb_project,
            entity="niklas-nertinger-university-of-oxford",
            name=run_name,
            config=vars(args),
            group=args.variant,
            tags=[args.variant, args.env_type]
        )
        use_wandb = True
    except Exception as e:
        print(f"W&B Initialization Failed: {e}. Running without W&B.")
    
    heartbeat_path = os.path.join(seed_dir, "heartbeat.csv")
    heartbeat_file = open(heartbeat_path, "w", newline="")
    heartbeat_writer = None

    if args.env_type == "metaworld":
        temp_env = MetaWorldWrapper(benchmark=args.mt_setting, seed=args.seed)
        num_tasks = temp_env.num_tasks
        temp_env.close()
    else:
        num_tasks = 5
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    def make_env(seed, idx, initial_task_idx=0, auto_cycle_task=False):
        def thunk():
            if args.env_type == "ComplexCartPole":
                env = ComplexCartPole()
            elif args.env_type == "IdenticalCartPole":
                env = IdenticalCartPole()
            elif args.env_type == "metaworld":
                env = MetaWorldWrapper(
                    benchmark=args.mt_setting, 
                    seed=seed, 
                    initial_task_idx=initial_task_idx, 
                    auto_cycle_task=auto_cycle_task
                )
            elif args.env_type == "MultiTaskLunarLander":
                env = MultiTaskLunarLander(task_idx=initial_task_idx)
            elif args.env_type == "IdenticalLunarLander":
                env = IdenticalLunarLander(task_idx=initial_task_idx)
            
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            if hasattr(env.observation_space, 'seed'):
                env.observation_space.seed(seed)
            return env
        return thunk

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.seed + i, 
                i, 
                initial_task_idx=i % num_tasks,
                auto_cycle_task=True
            ) 
            for i in range(args.num_envs)
        ]
    )
    
    if args.env_type == "ComplexCartPole":
        eval_env = ComplexCartPole()
    elif args.env_type == "IdenticalCartPole":
        eval_env = IdenticalCartPole()
    elif args.env_type == "metaworld":
        eval_env = MetaWorldWrapper(benchmark=args.mt_setting, seed=args.seed + 1000)
    elif args.env_type == "MultiTaskLunarLander":
        eval_env = MultiTaskLunarLander()
    elif args.env_type == "IdenticalLunarLander":
        eval_env = IdenticalLunarLander()
        
    eval_env.action_space.seed(args.seed + 1000)
    if hasattr(eval_env.observation_space, 'seed'):
        eval_env.observation_space.seed(args.seed + 1000)
    
    agent = DetActorCritic(
        envs, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        variant=args.variant,
        lora_rank=args.lora_rank,
        is_continuous=not isinstance(envs.single_action_space, gym.spaces.Discrete),
        use_task_embedding=False,
        num_tasks=num_tasks
    ).to(device)

    if args.compile:
        # Wrap strictly the VarShare backbone to minimize kernel launch overhead
        print("Compiling agent.actor_backbone via Torch 2.0 Triton...")
        agent.actor_backbone = torch.compile(agent.actor_backbone, mode="reduce-overhead")
    
    # Register Tasks
    for t_idx in range(num_tasks):
        agent.actor_backbone.add_task(t_idx, mu_init=args.mu_init)
        agent.critic_backbone.add_task(t_idx, mu_init=args.mu_init)
        agent.actor_head.add_task(t_idx, mu_init=args.mu_init)
        agent.critic_head.add_task(t_idx, mu_init=args.mu_init)
    
    actor_params = list(agent.actor_backbone.parameters()) + list(agent.actor_head.parameters())
    if hasattr(agent, 'actor_logstd'):
        actor_params += [agent.actor_logstd]
    critic_params = list(agent.critic_backbone.parameters()) + list(agent.critic_head.parameters())
    
    # We rely on explicit mu_l2_coef instead of optim weight_decay, to protect theta
    optimizer = optim.Adam([
        {'params': actor_params, 'lr': args.lr_actor},
        {'params': critic_params, 'lr': args.lr_critic}
    ], eps=1e-5)
    
    class DetVarSharePPO(PPO):
        def __init__(self, *args, batch_size=64, max_grad_norm=2.0, mu_l2_coef=0.001, variant="base", ara_coef=0.1, cagrad_c=0.4, routing_alpha=0.1, routing_lambda=0.001, **kwargs):
            super().__init__(*args, **kwargs)
            self._batch_size = batch_size
            self.max_grad_norm = max_grad_norm
            self.mu_l2_coef = mu_l2_coef
            self.variant = variant
            self.ara_coef = ara_coef
            self.cagrad_c = cagrad_c
            self.routing_alpha = routing_alpha
            self.routing_lambda = routing_lambda

        @property
        def batch_size(self):
            return self._batch_size

        def update_from_storage(self, obs, actions, logprobs, returns, advantages, values, task_ids=None):
            self.agent.train()
            inds = np.arange(len(obs))
            clipfracs = []
            td_vars = []
            
            for epoch in range(self.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, len(obs), self.batch_size):
                    end = start + self.batch_size
                    mb_inds = inds[start:end]
                    
                    mb_obs = obs[mb_inds]
                    mb_actions = actions[mb_inds]
                    mb_task_ids = task_ids[mb_inds] if task_ids is not None else None
                    mb_logprobs = logprobs[mb_inds]
                    mb_returns = returns[mb_inds]
                    mb_advantages = advantages[mb_inds]
                    
                    _, newlogprob, entropy, newvalue, ara_logits = self.agent.get_action_and_value(
                        mb_obs, mb_actions, task_idx=mb_task_ids
                    )
                    
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    td_vars.append((mb_returns - newvalue).var().item())

                    entropy_loss = entropy.mean()
                    
                    # Structural Regularization (L2, L1, or Decay)
                    reg_penalty = 0.0
                    
                    if self.variant == "decay":
                        # Progressively penalize mu more in deeper layers to force early sharing
                        num_layers = self.agent.actor_backbone.num_layers
                        for name, param in self.agent.named_parameters():
                            if "mus" in name:
                                # Extract layer idx if possible (e.g. `actor_backbone.layers.0.mus...`)
                                layer_idx = -1
                                parts = name.split(".")
                                for p in parts:
                                    if p.isdigit(): 
                                        layer_idx = int(p)
                                        break
                                
                                penalty_factor = 1.0
                                if layer_idx >= 0:
                                    # Example: Layer 0 -> scale 0.1, Layer 1 -> scale 1.0
                                    penalty_factor = (layer_idx + 1) / (num_layers + 1e-5)
                                    
                                reg_penalty += penalty_factor * torch.sum(param ** 2)
                                
                    else:
                        for name, param in self.agent.named_parameters():
                            if "mus" in name or "betas" in name:
                                if self.variant == "l1":
                                    reg_penalty += torch.sum(torch.abs(param))
                                else:
                                    reg_penalty += torch.sum(param ** 2)
                    
                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss + self.mu_l2_coef * reg_penalty

                    # ARA Loss
                    ara_loss_val = 0.0
                    if ara_logits is not None and mb_task_ids is not None:
                        import torch.nn.functional as F
                        ara_loss_tensor = F.cross_entropy(ara_logits, mb_task_ids)
                        loss += self.ara_coef * ara_loss_tensor
                        ara_loss_val = (self.ara_coef * ara_loss_tensor).item()

                    self.optimizer.zero_grad()
                    
                    if self.variant == "pcgrad" and mb_task_ids is not None:
                        # PCGrad: Compute per-task gradients and project conflicts
                        unique_tasks = torch.unique(mb_task_ids)
                        task_grads = []
                        
                        # 1. Gather per-task gradients
                        for t_id in unique_tasks:
                            self.optimizer.zero_grad()
                            t_mask = (mb_task_ids == t_id)
                            # Recompute loss ONLY for this task's items
                            # (A slight approximation for speed, pulling isolated components)
                            t_pg_loss1 = -mb_advantages[t_mask] * ratio[t_mask]
                            t_pg_loss2 = -mb_advantages[t_mask] * torch.clamp(ratio[t_mask], 1 - self.clip_coef, 1 + self.clip_coef)
                            t_pg_loss = torch.max(t_pg_loss1, t_pg_loss2).mean()
                            
                            t_v_loss = 0.5 * ((newvalue[t_mask].view(-1) - mb_returns[t_mask]) ** 2).mean()
                            t_entropy_loss = entropy[t_mask].mean()
                            
                            t_loss = t_pg_loss - self.ent_coef * t_entropy_loss + self.vf_coef * t_v_loss
                            
                            # Backward to populate .grad
                            t_loss.backward(retain_graph=True)
                            
                            # Flatten grads
                            flat_grad = []
                            for p in self.agent.parameters():
                                if p.grad is not None:
                                    flat_grad.append(p.grad.view(-1).clone())
                                else:
                                    flat_grad.append(torch.zeros_like(p.view(-1)))
                            task_grads.append(torch.cat(flat_grad))
                            
                        self.optimizer.zero_grad()
                        
                        # 2. Project Conflicting Gradients
                        num_tasks = len(task_grads)
                        if num_tasks > 1:
                            random.shuffle(task_grads)
                            for i in range(num_tasks):
                                grad_i = task_grads[i]
                                for j in range(num_tasks):
                                    if i == j: continue
                                    grad_j = task_grads[j]
                                    
                                    dot_prod = torch.dot(grad_i, grad_j)
                                    if dot_prod < 0:
                                        # Project i onto normal of j
                                        grad_i = grad_i - (dot_prod / (torch.norm(grad_j)**2 + 1e-8)) * grad_j
                                task_grads[i] = grad_i
                                
                            # Aggregate
                            final_grad = torch.stack(task_grads).sum(dim=0)
                            
                            # Apply back to parameters
                            offset = 0
                            for p in self.agent.parameters():
                                numel = p.numel()
                                p.grad = final_grad[offset:offset+numel].view_as(p).clone()
                                offset += numel
                        else:
                            # Fallback if only 1 task in mini-batch
                            loss.backward()
                    elif self.variant == "cagrad" and mb_task_ids is not None:
                        # CAGrad: Compute per-task gradients and project conflicts via dual solving
                        unique_tasks = torch.unique(mb_task_ids)
                        task_grads = []
                        
                        for t_id in unique_tasks:
                            self.optimizer.zero_grad()
                            t_mask = (mb_task_ids == t_id)
                            t_pg_loss1 = -mb_advantages[t_mask] * ratio[t_mask]
                            t_pg_loss2 = -mb_advantages[t_mask] * torch.clamp(ratio[t_mask], 1 - self.clip_coef, 1 + self.clip_coef)
                            t_pg_loss = torch.max(t_pg_loss1, t_pg_loss2).mean()
                            
                            t_v_loss = 0.5 * ((newvalue[t_mask].view(-1) - mb_returns[t_mask]) ** 2).mean()
                            t_entropy_loss = entropy[t_mask].mean()
                            
                            t_loss = t_pg_loss - self.ent_coef * t_entropy_loss + self.vf_coef * t_v_loss
                            t_loss.backward(retain_graph=True)
                            
                            flat_grad = []
                            for p in self.agent.parameters():
                                if p.grad is not None:
                                    flat_grad.append(p.grad.view(-1).clone())
                                else:
                                    flat_grad.append(torch.zeros_like(p.view(-1)))
                            task_grads.append(torch.cat(flat_grad))
                            
                        self.optimizer.zero_grad()
                        
                        num_tasks = len(task_grads)
                        if num_tasks > 1:
                            grads = torch.stack(task_grads)
                            g0 = grads.mean(dim=0)
                            GG = torch.mm(grads, grads.t()).cpu().numpy()
                            g0_norm = (g0.pow(2).sum() + 1e-8).sqrt().item()
                            
                            def obj(w):
                                gw_norm_sq = w.T.dot(GG).dot(w) + 1e-8
                                return w.T.dot(GG.sum(1) / num_tasks) + self.cagrad_c * g0_norm * np.sqrt(gw_norm_sq)
                                
                            from scipy.optimize import minimize
                            w0 = np.ones(num_tasks) / num_tasks
                            bnds = tuple((0.0, 1.0) for _ in w0)
                            cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0})
                            
                            res = minimize(obj, w0, bounds=bnds, constraints=cons)
                            w = res.x
                            
                            gw = (grads * torch.tensor(w, dtype=torch.float32, device=grads.device).view(-1, 1)).sum(dim=0)
                            gw_norm = (gw.pow(2).sum() + 1e-8).sqrt()
                            
                            final_grad = g0 + self.cagrad_c * g0_norm * gw / gw_norm
                            
                            offset = 0
                            for p in self.agent.parameters():
                                numel = p.numel()
                                p.grad = final_grad[offset:offset+numel].view_as(p).clone()
                                offset += numel
                        else:
                            loss.backward()
                    elif self.variant == "routing" and mb_task_ids is not None:
                        unique_tasks = torch.unique(mb_task_ids)
                        
                        # Identify strict backbone parameters
                        shared_params = []
                        for name, p in self.agent.named_parameters():
                            if ".mus." not in name:
                                shared_params.append(p)
                                
                        task_shared_grads_list = []
                        
                        for t_id in unique_tasks:
                            self.optimizer.zero_grad()
                            t_mask = (mb_task_ids == t_id)
                            t_pg_loss1 = -mb_advantages[t_mask] * ratio[t_mask]
                            t_pg_loss2 = -mb_advantages[t_mask] * torch.clamp(ratio[t_mask], 1 - self.clip_coef, 1 + self.clip_coef)
                            t_pg_loss = torch.max(t_pg_loss1, t_pg_loss2).mean()
                            
                            t_v_loss = 0.5 * ((newvalue[t_mask].view(-1) - mb_returns[t_mask]) ** 2).mean()
                            t_entropy_loss = entropy[t_mask].mean()
                            
                            t_loss = t_pg_loss - self.ent_coef * t_entropy_loss + self.vf_coef * t_v_loss
                            t_loss.backward(retain_graph=True)
                            
                            grads_dict = {}
                            for p in shared_params:
                                if p.grad is not None:
                                    grads_dict[p] = p.grad.clone()
                                else:
                                    grads_dict[p] = torch.zeros_like(p)
                            task_shared_grads_list.append(grads_dict)
                            
                        self.optimizer.zero_grad()
                        
                        num_tasks = len(task_shared_grads_list)
                        if num_tasks > 1:
                            # Keep pristine copies for the residual r_t
                            raw_shared_grads = [{p: g.clone() for p, g in d.items()} for d in task_shared_grads_list]
                            
                            def dot_prod_dict(d1, d2):
                                return sum(torch.sum(d1[p] * d2[p]) for p in d1)
                            def norm_sq_dict(d):
                                return sum(torch.sum(d[p]**2) for p in d)
                                
                            indices = list(range(num_tasks))
                            random.shuffle(indices)
                            for i in indices:
                                grad_i = task_shared_grads_list[i]
                                for j in indices:
                                    if i == j: continue
                                    grad_j = task_shared_grads_list[j]
                                    
                                    dot_prod = dot_prod_dict(grad_i, grad_j)
                                    if dot_prod < 0:
                                        norm_j_sq = norm_sq_dict(grad_j) + 1e-8
                                        scale = dot_prod / norm_j_sq
                                        for p in grad_i:
                                            grad_i[p] = grad_i[p] - scale * grad_j[p]
                                task_shared_grads_list[i] = grad_i
                                
                            # Aggregate final shared \bar{g}_\theta
                            bar_g_theta = {}
                            for p in shared_params:
                                bar_g_theta[p] = sum(d[p] for d in task_shared_grads_list) / num_tasks
                                p.grad = bar_g_theta[p].clone()
                                
                            # Route Residuals down to adapter components
                            for idx, t_id in enumerate(unique_tasks):
                                t_key = str(t_id.item())
                                raw_g_t = raw_shared_grads[idx]
                                
                                # Iterate over modules to find DetVarShareLayers
                                from src.models_routing import DetVarShareLayer
                                for m in self.agent.modules():
                                    if isinstance(m, DetVarShareLayer):
                                        if t_key in m.mus:
                                            # Ensure theta participated in gradients
                                            if m.theta in bar_g_theta:
                                                r_t = raw_g_t[m.theta] - bar_g_theta[m.theta]
                                                # \bar{g}_{\mu_t} = \alpha r_t + \lambda \mu_t
                                                m.mus[t_key].grad = self.routing_alpha * r_t + self.routing_lambda * m.mus[t_key].data
                        else:
                            loss.backward()
                    else:
                        # Standard Backward
                        loss.backward()
                    
                    total_norm = 0.0
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
            return {
                "loss": loss.item(),
                "policy_loss": pg_loss.item(),
                "value_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "reg_penalty": reg_penalty.item() * self.mu_l2_coef if isinstance(reg_penalty, torch.Tensor) else reg_penalty * self.mu_l2_coef,
                "ara_loss": ara_loss_val,
                "grad_norm": total_norm,
                "clip_fraction": np.mean(clipfracs),
                "td_error_variance": np.mean(td_vars) if td_vars else 0.0,
                "rollout_len": len(obs)
            }

    ppo = DetVarSharePPO(
        agent, 
        optimizer, 
        lr=args.lr_actor, 
        gamma=args.gamma, 
        gae_lambda=args.gae_lambda, 
        clip_coef=args.eps_clip, 
        ent_coef=args.ent_coef, 
        update_epochs=args.k_epochs, 
        minibatch_size=args.batch_size,
        device=device,
        max_grad_norm=args.max_grad_norm,
        mu_l2_coef=args.mu_l2_coef,
        variant=args.variant,
        cagrad_c=args.cagrad_c,
        routing_alpha=args.routing_alpha,
        routing_lambda=args.routing_lambda
    )
    
    n_steps = args.n_steps
    n_updates = args.total_timesteps // (n_steps * args.num_envs)
    
    task_idx = 0
    # eval_env doesn't auto-rotate so we set it
    eval_env.reset_task(task_idx)

    obs, info = envs.reset(seed=args.seed)
    
    # Track the active task ID for each environment
    current_task_ids = np.array([args.task_id if args.algo == "oracle" else i % num_tasks for i in range(args.num_envs)], dtype=np.int64)
    
    task_reward_window = {t: deque(maxlen=25) for t in range(num_tasks)}
    task_success_window = {t: deque(maxlen=25) for t in range(num_tasks)}
    time_window = deque(maxlen=25)
    
    num_episodes_finished = 0
    
    history = []
    start_time = time.time()
    global_step = 0
    next_eval_step = 0
    eval_reward_current = 0.0
    eval_metrics = {}
    
    last_k_rewards = deque(maxlen=3)
    start_update = 0
    
    checkpoint_path = os.path.join(seed_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_update = checkpoint["update"] + 1
        global_step = checkpoint["global_step"]
        num_episodes_finished = checkpoint["num_episodes_finished"]
        next_eval_step = checkpoint["next_eval_step"]
        eval_reward_current = checkpoint["eval_reward_current"]
        eval_metrics = checkpoint.get("eval_metrics", {})
        if "task_reward_window" in checkpoint:
            task_reward_window = checkpoint["task_reward_window"]
        if "task_success_window" in checkpoint:
            task_success_window = checkpoint["task_success_window"]
        if "current_task_ids" in checkpoint:
            current_task_ids = checkpoint["current_task_ids"]
        if "time_window" in checkpoint:
            time_window.extend(checkpoint["time_window"])
        if "last_k_rewards" in checkpoint:
            last_k_rewards.extend(checkpoint["last_k_rewards"])
        
        # Load history json if it exists to append to it
        hist_path = os.path.join(seed_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                history = json.load(f)
    
    for update in range(start_update, n_updates):
        mb_obs = torch.zeros((n_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        mb_actions = torch.zeros((n_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        mb_logprobs = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_rewards = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_dones = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_values = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_task_ids = torch.zeros((n_steps, args.num_envs), dtype=torch.long).to(device)
        
        for step in range(n_steps):
            global_step += 1 * args.num_envs
            
            with torch.no_grad():
                task_ids_tensor = torch.tensor(current_task_ids, dtype=torch.long).to(device)
                action, logprob, _, value, _ = agent.get_action_and_value(
                    torch.tensor(obs).float().to(device), 
                    task_idx=task_ids_tensor
                )
                
            mb_obs[step] = torch.tensor(obs).float().to(device)
            mb_actions[step] = action
            mb_logprobs[step] = logprob
            mb_values[step] = value.flatten()
            mb_task_ids[step] = task_ids_tensor
            
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            dones = np.logical_or(terminations, truncations)
            mb_dones[step] = torch.tensor(dones).float().to(device)
            mb_rewards[step] = torch.tensor(rewards).float().to(device)
            
            obs = next_obs
            
            if "episode" in infos:
                if "_episode" in infos:
                    for i, has_ep in enumerate(infos["_episode"]):
                        if has_ep:
                            num_episodes_finished += 1
                            t = current_task_ids[i]
                            task_reward_window[t].append(infos["episode"]["r"][i])
                            time_window.append(infos["episode"]["l"][i])
                            if "success" in infos:
                                task_success_window[t].append(infos["success"][i])
            elif "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        num_episodes_finished += 1
                        t = info.get("task_idx", current_task_ids[i])
                        task_reward_window[t].append(info["episode"]["r"])
                        time_window.append(info["episode"]["l"])
                        if "success" in info:
                            task_success_window[t].append(info["success"])
            
            # Update current task IDs for any env that auto-cycled / reset
            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "task_idx" in info:
                        current_task_ids[i] = info["task_idx"]
            elif "episode" in infos:
                if "_episode" in infos:
                    for i, has_ep in enumerate(infos["_episode"]):
                        if has_ep:
                            if args.algo != "oracle" and args.env_type == "metaworld":
                                current_task_ids[i] = (current_task_ids[i] + 1) % num_tasks

        with torch.no_grad():
            task_ids_tensor = torch.tensor(current_task_ids, dtype=torch.long).to(device)
            _, _, _, next_value, _ = agent.get_action_and_value(
                torch.tensor(obs).float().to(device), 
                task_idx=task_ids_tensor
            )
            next_value = next_value.reshape(1, -1)
            
        advantages = torch.zeros_like(mb_rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - mb_dones[t]
                nextvalues = next_value.flatten()
            else:
                nextnonterminal = 1.0 - mb_dones[t]
                nextvalues = mb_values[t+1]
            
            delta = mb_rewards[t] + args.gamma * nextvalues * nextnonterminal - mb_values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + mb_values
        
        b_obs = mb_obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = mb_logprobs.reshape(-1)
        b_actions = mb_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = mb_values.reshape(-1)
        b_task_ids = mb_task_ids.reshape(-1)
        
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-7)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-7)
        
        metrics = ppo.update_from_storage(
            b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values, task_ids=b_task_ids
        )
        
        explained_var = calculate_explained_variance(b_returns.cpu().numpy(), b_values.cpu().numpy())
        
        # Aggregate Architecture metrics per-task
        arch_data = {}
        layer_mu_sums = collections.defaultdict(float)
        
        for t in range(num_tasks):
            t_data = agent.actor_backbone.get_metrics(t)
            if t_data:
                for k, v in t_data.items():
                    # k is like "layer0_norm_mu"
                    if "norm_mu" in k:
                        arch_data[f"task_{t}_{k}"] = v  # e.g. task_0_layer0_norm_mu
                        layer_mu_sums[k] += v
                    elif "norm_theta" in k:
                        if t == 0:
                            arch_data[k] = v  # e.g. layer0_norm_theta
        
        # Per layer averages across tasks
        for k, v_sum in layer_mu_sums.items():
            arch_data[k] = v_sum / num_tasks  # e.g. layer0_norm_mu
            layer_name = k.split("_")[0]
            layer_theta = arch_data.get(f"{layer_name}_norm_theta", 0.0)
            layer_mu_avg = arch_data[k]
            arch_data[f"{layer_name}_sharing_ratio"] = layer_theta / (layer_theta + layer_mu_avg + 1e-8)
                    
        # Add global structural metrics (variance and sparsity)
        global_arch_data = agent.actor_backbone.get_global_metrics()
        for k, v in global_arch_data.items():
            arch_data[k] = v
        
        # Average these out globally for history:
        mean_norm_mu = np.mean([v for k,v in arch_data.items() if "norm_mu" in k and "task_" not in k]) if arch_data else 0.0
        mean_norm_theta = np.mean([v for k,v in arch_data.items() if "norm_theta" in k and "task_" not in k]) if arch_data else 0.0
        sharing_ratio = mean_norm_theta / (mean_norm_theta + mean_norm_mu + 1e-8)
        mean_adapter_sparsity = np.mean([v for k,v in arch_data.items() if "adapter_sparsity" in k]) if arch_data else 0.0
        mean_adapter_variance = np.mean([v for k,v in arch_data.items() if "adapter_variance" in k]) if arch_data else 0.0
        
        # Average per-task train rewards
        for t in range(num_tasks):
            eval_metrics[f"performance/train_reward_task_{t}"] = np.mean(task_reward_window[t]) if task_reward_window[t] else 0.0
            eval_metrics[f"performance/train_success_task_{t}"] = np.mean(task_success_window[t]) if task_success_window[t] else 0.0
            
        all_train_rewards = [r for d in task_reward_window.values() for r in d]
        all_train_successes = [s for d in task_success_window.values() for s in d]
        avg_reward = np.mean(all_train_rewards) if all_train_rewards else 0.0
        avg_success = np.mean(all_train_successes) if all_train_successes else 0.0
        
        hist_entry = {
            "step": global_step,
            "reward": avg_reward,
            "success": avg_success,
            "norm_mu": mean_norm_mu,
            "norm_theta": mean_norm_theta,
            "sharing_ratio": sharing_ratio,
            "loss_total": metrics["loss"],
            "loss_policy": metrics["policy_loss"],
            "loss_value": metrics["value_loss"],
            "loss_entropy": metrics["entropy"],
            "loss_reg_penalty": metrics["reg_penalty"],
            "loss_ara": metrics["ara_loss"],
            "explained_variance": explained_var,
        }
        
        if arch_data:
            for k, v in arch_data.items():
                hist_entry[k] = v
        
        history.append(hist_entry)
        
        sps = int(global_step / (time.time() - start_time))
        
        if args.eval_mode and global_step >= next_eval_step:
            print(f"\n>>> Running Evaluation at Step {global_step}...")
            task_stats = evaluate(agent, eval_env, device, num_episodes=args.eval_episodes, num_tasks=num_tasks)
            
            eval_reward_mean = np.mean([s["reward"] for s in task_stats.values()])
            eval_success_mean = np.mean([s["success"] for s in task_stats.values()])
            
            for k, v in task_stats.items():
                eval_metrics[f"eval/reward_task_{k}"] = v["reward"]
                eval_metrics[f"eval/success_task_{k}"] = v["success"]
                
            eval_metrics["eval/mean_reward"] = eval_reward_mean
            eval_metrics["eval/mean_success"] = eval_success_mean
            eval_reward_current = eval_reward_mean
            
            # Ablation: Evaluate using Shared Backbone Only (zero out adapters)
            print(">>> Running Ablation: Shared-Backbone Only Evaluation...")
            original_state = {k: v.clone() for k, v in agent.state_dict().items()}
            # Zero out adapter parameters
            for name, param in agent.named_parameters():
                if "mus" in name or "betas" in name or "gammas" in name or "mus_A" in name or "mus_B" in name:
                    param.data.zero_()
            
            ablation_stats = evaluate(agent, eval_env, device, num_episodes=args.eval_episodes, num_tasks=num_tasks)
            eval_metrics["eval/shared_backbone_only_reward"] = np.mean([s["reward"] for s in ablation_stats.values()])
            eval_metrics["eval/shared_backbone_only_success"] = np.mean([s["success"] for s in ablation_stats.values()])
            print(f"Shared-Backbone Reward: {eval_metrics['eval/shared_backbone_only_reward']:.2f} | Success: {eval_metrics['eval/shared_backbone_only_success']:.2f}")
            
            # Restore model
            agent.load_state_dict(original_state)
            
            last_k_rewards.append(eval_reward_mean)
            avg_last_k = np.mean(last_k_rewards)
            
            print(f"Eval Reward: {eval_reward_mean:.2f} | Eval Success: {eval_success_mean:.2f}")
            print(f"FINAL_EVAL_SCORE: {eval_success_mean:.4f}")
            print(f"FINAL_EVAL_REWARD: {avg_last_k:.4f}")
            next_eval_step += args.eval_freq
            
            if report_callback:
                report_callback(eval_reward_mean, global_step)
            
        history[-1]["eval_reward"] = eval_reward_current
        history[-1]["eval_success"] = eval_metrics.get("eval/mean_success")
        history[-1]["grad_norm"] = metrics["grad_norm"]
        history[-1]["episodes"] = num_episodes_finished
        
        full_metrics = {
            "TOTAL_ENV_STEPS": global_step,
            "EPISODES_COMPLETED": num_episodes_finished,
            "TOTAL_GRAD_STEPS": (update + 1) * args.k_epochs * (n_steps * args.num_envs // args.batch_size),
            "WALL_CLOCK_TIME": time.time() - start_time,
            "SPS": sps,
            "loss/total": metrics["loss"],
            "loss/policy": metrics["policy_loss"],
            "loss/value": metrics["value_loss"],
            "loss/entropy": metrics["entropy"],
            "loss/reg_penalty": metrics["reg_penalty"],
            "loss/ara": metrics["ara_loss"],
            "diagnostics/grad_norm": metrics["grad_norm"],
            "diagnostics/clip_fraction": metrics["clip_fraction"],
            "diagnostics/explained_variance": explained_var,
            "diagnostics/td_error_variance": metrics.get("td_error_variance", 0.0),
            "detvarshare/mean_norm_mu": mean_norm_mu,
            "detvarshare/mean_norm_theta": mean_norm_theta,
            "detvarshare/sharing_ratio": sharing_ratio,
            "detvarshare/adapter_sparsity": mean_adapter_sparsity,
            "detvarshare/adapter_variance": mean_adapter_variance,
            "performance/train_mean_reward": avg_reward,
            "performance/train_mean_success": avg_success,
        }
        for t in range(num_tasks):
            eval_metrics[f"performance/train_reward_task_{t}"] = eval_metrics.get(f"performance/train_reward_task_{t}", 0.0)
            eval_metrics[f"performance/train_success_task_{t}"] = eval_metrics.get(f"performance/train_success_task_{t}", 0.0)
            eval_metrics[f"eval/reward_task_{t}"] = eval_metrics.get(f"eval/reward_task_{t}", 0.0)
            eval_metrics[f"eval/success_task_{t}"] = eval_metrics.get(f"eval/success_task_{t}", 0.0)
        
        full_metrics.update(eval_metrics)
        # Flatten arch_data into full_metrics
        for k, v in arch_data.items():
            full_metrics[f"arch/{k}"] = v
        
        if heartbeat_writer is None:
            heartbeat_writer = csv.DictWriter(heartbeat_file, fieldnames=list(full_metrics.keys()))
            heartbeat_writer.writeheader()
            
        heartbeat_writer.writerow(full_metrics)
        heartbeat_file.flush()
        
        if use_wandb:
            try:
                wandb.log(full_metrics, step=global_step)
            except Exception as e:
                print(f"W&B Log Failed: {e}. Disabling W&B for remainder of run.")
                use_wandb = False
        
        print_freq = 2500
        step_increment = n_steps * args.num_envs
        if global_step // print_freq > (global_step - step_increment) // print_freq:
             print(f"Step {global_step:>7d} | Eps {num_episodes_finished:>4d} | Rew {avg_reward:>6.1f} | SPS {sps:>4d} | Loss {metrics['loss']:>7.3f} | Ent {metrics['entropy']:>5.3f}")

        # Periodic State-Saving
        if args.eval_mode:
            checkpoint_interval = max(1, n_updates // 25)
            if update > 0 and update % checkpoint_interval == 0:
                print(f"[CHECKPOINT] Saving periodic state at Step {global_step}...")
                chkpt = {
                    "update": update,
                    "global_step": global_step,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(chkpt, os.path.join(seed_dir, f"checkpoint_step_{global_step}.pt"))

        if shutdown_requested:
            print(f"\n[CHECKPOINT] Saving state at update {update} (Step {global_step})...")
            checkpoint = {
                "update": update,
                "global_step": global_step,
                "num_episodes_finished": num_episodes_finished,
                "next_eval_step": next_eval_step,
                "eval_reward_current": eval_reward_current,
                "eval_metrics": eval_metrics,
                "task_reward_window": task_reward_window,
                "task_success_window": task_success_window,
                "time_window": list(time_window),
                "last_k_rewards": list(last_k_rewards),
                "current_task_ids": current_task_ids,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Save history incrementally
            with open(os.path.join(seed_dir, "history.json"), "w") as f:
                json.dump(history, f, cls=NumpyEncoder)
                
            print("[CHECKPOINT] Safe shutdown complete. Exiting with code 99.")
            sys.exit(99)

    total_duration = time.time() - start_time
    print(f"\nFinal Average Reward: {avg_reward:.2f}")
    print(f"Total Duration: {total_duration:.2f}s")
    
    print(f">>> Generating scientific plots in {seed_dir}...")
    def plot_metric(data, key, ylabel, filename):
        plt.figure(figsize=(10, 6))
        steps = [d["step"] for d in data]
        values = [d[key] for d in data]
        plt.plot(steps, values)
        plt.xlabel("Total Environment Steps")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Steps ({args.exp_name})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(seed_dir, filename))
        plt.close()

    plot_metric(history, "reward", "Training Reward (Moving Avg)", "train_reward.png")
    plot_metric(history, "success", "Training Success Rate", "train_success.png")
    plot_metric(history, "norm_mu", "Mean Norm Mu", "norm_mu.png")
    plot_metric(history, "norm_theta", "Mean Norm Theta", "norm_theta.png")
    plot_metric(history, "sharing_ratio", "Sharing Ratio", "sharing_ratio.png")
    plot_metric(history, "episodes", "Total Episodes", "episodes.png")
    
    eval_history = [d for d in history if d.get("eval_reward") is not None]
    if eval_history:
         plot_metric(eval_history, "eval_reward", "Evaluation Reward", "eval_reward.png")
         plot_metric(eval_history, "eval_success", "Evaluation Success Rate", "eval_success.png")
    plot_metric(history, "grad_norm", "Gradient Norm", "grad_norm.png")

    # Stacked Loss Plot (Total, L2, Value, Policy, Entropy)
    try:
        import pandas as pd
        def smooth(arr):
            return pd.Series(arr).rolling(window=50, min_periods=1).mean().tolist()
            
        plt.figure(figsize=(12, 10))
        steps = [h["step"] for h in history]
        
        plt.subplot(2, 2, 1)
        plt.plot(steps, smooth([h["loss_total"] for h in history]), label="Total Loss", color="black")
        plt.title("Total Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(steps, smooth([h["loss_entropy"] for h in history]), label="Entropy Loss", color="red")
        plt.title("Entropy / Exploration")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(steps, smooth([h["loss_value"] for h in history]), label="Value Loss", color="orange")
        plt.plot(steps, smooth([h["loss_policy"] for h in history]), label="Policy Loss", color="green")
        plt.title("RL Objectives")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(steps, smooth([h["loss_reg_penalty"] for h in history]), label="Reg Penalty", color="blue")
        if any(h["loss_ara"] > 0 for h in history):
            plt.plot(steps, smooth([h["loss_ara"] for h in history]), label="ARA Loss", color="purple")
        plt.title("Structural / Regularization Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle(f"Optimization Landscape ({args.exp_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(seed_dir, "loss_breakdown.png"))
        plt.close()
    except Exception as e:
        print(f"Failed to plot loss breakdown: {e}")

    # Layer-wise Dynamics Plots
    try:
        # Identify dynamic layer keys
        layer_mu_keys = sorted([k for k in history[0].keys() if "norm_mu" in k and k.startswith("layer")])
        layer_theta_keys = sorted([k for k in history[0].keys() if "norm_theta" in k and k.startswith("layer")])
        
        steps = [h["step"] for h in history]

        # 1. Standard plot for Mu
        if layer_mu_keys:
            plt.figure(figsize=(10, 6))
            for key in layer_mu_keys:
                layer_idx = key.split("_")[0].replace("layer", "")
                plt.plot(steps, [h[key] for h in history], label=f"Layer {layer_idx} (Mu)")
                
            plt.xlabel("Total Environment Steps")
            plt.ylabel("Mu Norm Magnitude")
            plt.title(f"Layer-wise Structural Evolution: Mu ({args.exp_name})")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(seed_dir, "layerwise_mu_norms.png"))
            plt.close()

        # 2. Standard plot for Theta
        if layer_theta_keys:
            plt.figure(figsize=(10, 6))
            for key in layer_theta_keys:
                layer_idx = key.split("_")[0].replace("layer", "")
                plt.plot(steps, [h[key] for h in history], label=f"Layer {layer_idx} (Theta)")
                
            plt.xlabel("Total Environment Steps")
            plt.ylabel("Theta Norm Magnitude")
            plt.title(f"Layer-wise Structural Evolution: Theta ({args.exp_name})")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(seed_dir, "layerwise_theta_norms.png"))
            plt.close()
    except Exception as e:
        print(f"Failed to plot layer-wise dynamics: {e}")

if __name__ == "__main__":
    train()
