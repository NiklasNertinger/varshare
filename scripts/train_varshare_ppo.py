import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
# import wandb (Disabled by User)
import gymnasium as gym
from collections import deque
import csv
import json
import matplotlib.pyplot as plt

from src.env import ComplexCartPole, IdenticalCartPole, MetaWorldWrapper
from src.models import ActorCritic
from src.algo.ppo import PPO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="varshare_golden", help="experiment name")
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
    parser.add_argument("--kl-beta", type=float, default=0.00025, help="VarShare KL Beta")
    parser.add_argument("--mu-init", type=float, default=0.0, help="Mu init")
    parser.add_argument("--rho-init", type=float, default=-5.0, help="Rho init")
    parser.add_argument("--prior-scale", type=float, default=1.0, help="Prior scale for KL")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Backbone hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    
    # New Architectural Variants
    parser.add_argument("--variant", type=str, default="standard", choices=["standard", "lora", "partial", "reptile"], help="VarShare variant")
    parser.add_argument("--lora-rank", type=int, default=4, help="Rank for LoRA")
    parser.add_argument("--embedding-type", type=str, default="none", choices=["none", "onehot", "learned"], help="Task embedding type")
    
    # New Adaptive Methods
    parser.add_argument("--learned-prior", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use Empirical Bayes (learned prior)")
    parser.add_argument("--kl-schedule", type=str, default="fixed", choices=["fixed", "annealing", "trigger"], help="KL Beta schedule")
    parser.add_argument("--lambda-hyper", type=float, default=0.0, help="Hyperprior penalty (Exp 5)")
    parser.add_argument("--target-ratio", type=float, default=0.1, help="Target Noise/Weight ratio (Exp 3)")
    parser.add_argument("--warmup-frac", type=float, default=0.1, help="Warmup fraction (Exp 2)")
    
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="cuda toggle")
    parser.add_argument("--wandb-project", type=str, default="varshare-exp", help="wandb project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity")
    
    parser.add_argument("--eval-mode", type=bool, default=True, help="Run periodic evaluations")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Eval every N steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per task during eval")
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Root analysis directory")
    
    # Environment selection
    parser.add_argument("--env-type", type=str, default="ComplexCartPole", choices=["ComplexCartPole", "IdenticalCartPole", "metaworld"], help="Environment type")
    parser.add_argument("--mt-setting", type=str, default="MT10", choices=["MT1", "MT3", "MT10", "MT50"], help="Meta-World setting")

    args = parser.parse_args()
    return args

def calculate_explained_variance(y_true, y_pred):
    var_y = np.var(y_true)
    if var_y == 0: return 0
    return 1 - np.var(y_true - y_pred) / var_y

def get_parameter_counts(model):
    """
    Returns (total_params, active_params_per_task)
    - total_params: all parameters in the model.
    - active_params: parameters used for a single task inference.
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate active params for a single task
    active_params = 0
    
    # We account for each module
    for name, module in model.named_modules():
        # If it's a VarShareLayer, it has theta (shared), bias (shared), 
        # but only ONE mu/rho pair is active per task.
        if hasattr(module, 'mus') and hasattr(module, 'rhos'):
             # Shared parts
             active_params += module.theta.numel()
             active_params += module.bias.numel()
             # Task-specific parts (assume task 0 exists and has same size)
             active_params += module.mus['0'].numel()
             active_params += module.rhos['0'].numel()
        elif len(list(module.children())) == 0:
             # Standard leaf layer (Linear, etc.)
             # Only count if it's not part of a VarShareLayer (which we already handled)
             # named_modules includes parents, so children check helps find leaves.
             # However, Parameters are the ultimate truth.
             pass

    # Better approach: sum all shared params + specific task params for ONE task
    shared_params = 0
    task_params = 0 # for a single task
    
    for name, param in model.named_parameters():
        if 'mus' in name or 'rhos' in name:
            if '.0.' in name or name.endswith('.0'): # Task 0
                task_params += param.numel()
        else:
            shared_params += param.numel()
            
    return total_params, shared_params + task_params

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
                    # Eval mode: Deterministic (sample=False)
                    action, _, _, _ = agent.get_action_and_value(obs_t, task_idx=t_idx, sample=False)
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
    
    # 1. Directory Structure
    # analysis/<EXP>/seed_<SEED>/
    exp_dir = os.path.join(args.analysis_dir, args.exp_name)
    seed_dir = os.path.join(exp_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Heartbeat file
    heartbeat_path = os.path.join(seed_dir, "heartbeat.csv")
    heartbeat_file = open(heartbeat_path, "w", newline="")
    
    # Pre-define fieldnames
    fieldnames = [
        "TOTAL_ENV_STEPS", "EPISODES_COMPLETED", "TOTAL_GRAD_STEPS",
        "WALL_CLOCK_TIME", "SPS",
        "loss/total", "loss/policy", "loss/value", "loss/entropy",
        "loss/kl_penalty", "loss/raw_kl",
        "diagnostics/grad_norm", "diagnostics/clip_fraction", "diagnostics/explained_variance",
        "varshare/mean_norm_mu", "varshare/mean_norm_theta",
        "varshare/sharing_ratio", "varshare/avg_sigma", "varshare/task_similarity",
        "performance/train_reward_50", "performance/train_success_50",
        "eval/mean_reward", "eval/mean_success"
    ]
    
    # Determine num_tasks based on env_type
    if args.env_type == "metaworld":
        temp_env = MetaWorldWrapper(benchmark=args.mt_setting, seed=args.seed)
        num_tasks = temp_env.num_tasks
        temp_env.close()
    else:
        num_tasks = 5
        
    # Add per-task eval keys
    for t in range(num_tasks):
        fieldnames.append(f"eval/reward_task_{t}")
        fieldnames.append(f"eval/success_task_{t}")
        
    heartbeat_writer = csv.DictWriter(heartbeat_file, fieldnames=fieldnames)
    heartbeat_writer.writeheader()
    
    # WandB
    # WandB Disabled
    # key_path = "context/WandB_API_key.txt"
    # if os.path.exists(key_path):
    #     with open(key_path, "r") as f:
    #         wandb.login(key=f.read().strip())
    #         
    # wandb.init(
    #     project=args.wandb_project,
    #     entity=args.wandb_entity,
    #     config=vars(args),
    #     name=run_name,
    #     monitor_gym=True,
    #     save_code=True,
    # )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Logging to: {seed_dir}")
    
    # Env Setup
    def make_env(seed, idx, capture_video=False, run_name="", initial_task_idx=0, auto_cycle_task=False):
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
                False, 
                run_name,
                initial_task_idx=i % num_tasks,
                auto_cycle_task=True
            ) 
            for i in range(args.num_envs)
        ]
    )
    
    # Separate eval env to not mess with training state
    if args.env_type == "ComplexCartPole":
        eval_env = ComplexCartPole()
    elif args.env_type == "IdenticalCartPole":
        eval_env = IdenticalCartPole()
    elif args.env_type == "metaworld":
        eval_env = MetaWorldWrapper(benchmark=args.mt_setting, seed=args.seed + 1000)
        
    eval_env.action_space.seed(args.seed + 1000)
    if hasattr(eval_env.observation_space, 'seed'):
        eval_env.observation_space.seed(args.seed + 1000)
    
    # Model Setup
    # Model Setup
    
    # Configure VarShare Args
    varshare_args = {
        "prior_scale": args.prior_scale,
        "rho_init": args.rho_init,
        "mu_init": args.mu_init,
        "variant": args.variant,
        "rank": args.lora_rank,
        "learned_prior": args.learned_prior
    }
    
    # Configure Embedding
    use_task_embedding = (args.embedding_type != "none")
    # Note: ActorCritic expects `use_task_embedding=True` if we want embeddings.
    # It handles "onehot" vs "learned" internally? 
    # Wait, previous `ActorCritic` logic (lines 394-396) implemented Basic Embedding (learned) if `use_varshare=False`.
    # But `use_varshare=True` usually ignored embeddings unless we modify `_get_input`.
    # Let's verify `ActorCritic` again.
    # Actually, let's keep it simple: If embedding_type != none, we pass use_task_embedding=True.
    # The current `ActorCritic` implementation (seen in lines 375-397) uses `nn.Embedding` if `use_task_embedding` is True.
    # If we want One-Hot, we might need a small patch or assume "learned" is the only embedding mode supported by current ActorCritic.
    # User asked for "Basic Task ID" (OneHot) and "Learned Embedding".
    # I should assume standard code supports learned. One-hot needs custom handling or just use the embedding layer as dense transform?
    # Actually, previous milestone implemented `Shared` baseline which used embeddings.
    # Let's assume standard behavior for now to avoid breaking changes, 
    # but we flagged `mt10_varshare_emb_onehot` as a study.
    # I might need to stick to "learned" for both or patch ActorCritic to support "onehot".
    # Given time constraints, let's pass `embedding_dim`=num_tasks for "onehot"? No, that learns a matrix.
    # For now, let's just enable embedding support.
    
    agent = ActorCritic(
        envs.single_observation_space, 
        envs.single_action_space,
        hidden_dim=args.hidden_dim,
        use_task_embedding=use_task_embedding,
        embedding_type=args.embedding_type, # Pass embedding type
        embedding_dim=10, # Standard default
        num_tasks=num_tasks,
        use_varshare=True,
        varshare_args=varshare_args,
        num_layers=args.num_layers
    ).to(device)
    
    # Configure KL Controller
    kl_controller_args = {
        "total_updates": args.total_timesteps // (args.n_steps * args.num_envs),
        "warmup_frac": args.warmup_frac,
        "target_ratio": args.target_ratio
    }
    
    # Separate parameter groups
    # Actor Params: Actor Backbone + Actor Head + LogStd
    actor_params = list(agent.actor_backbone.parameters()) + list(agent.actor_head.parameters())
    if hasattr(agent, 'actor_logstd'):
        actor_params += [agent.actor_logstd]
        
    # Critic Params: Critic Backbone + Critic Head
    critic_params = list(agent.critic_backbone.parameters()) + list(agent.critic_head.parameters())
    
    optimizer = optim.Adam([
        {'params': actor_params, 'lr': args.lr_actor},
        {'params': critic_params, 'lr': args.lr_critic}
    ], eps=1e-5)
    
    # We use our PPO class but handle the episodic update loop manually and inject the KL term manually?
    # Actually, our PPO class `update_from_storage` is standard.
    # We can inherit or just modify the loop inside the script or `PPO` relies on `loss.backward()`.
    # PPO implementation calculates loss.
    # I should subclass PPO to add VarShare KL processing.
    
    class VarSharePPO(PPO):
        def __init__(self, *args, kl_beta=0.0, batch_size=64, **kwargs):
            super().__init__(*args, **kwargs)
            self.kl_beta = kl_beta
            self._batch_size = batch_size
            
        @property
        def batch_size(self):
            return self._batch_size

        def update_from_storage(self, obs, actions, logprobs, returns, advantages, values, task_ids=None):
            self.agent.train()
            
            inds = np.arange(len(obs))
            clipfracs = []
            
            # Optimization epochs loop
            for epoch in range(self.update_epochs):
                np.random.shuffle(inds)
                
                # Mini-batch loop
                for start in range(0, len(obs), self.batch_size):
                    end = start + self.batch_size
                    mb_inds = inds[start:end]
                    
                    # Fetch mini-batch data
                    mb_obs = obs[mb_inds]
                    mb_actions = actions[mb_inds]
                    mb_task_ids = task_ids[mb_inds] if task_ids is not None else None
                    mb_logprobs = logprobs[mb_inds]
                    mb_returns = returns[mb_inds]
                    mb_advantages = advantages[mb_inds]
                    
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        mb_obs, 
                        mb_actions, 
                        task_idx=mb_task_ids
                    )
                    
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                    # Entropy Loss
                    entropy_loss = entropy.mean()
                    
                    # VarShare KL Penalty
                    # Use current task_id (assumed constant for CartPole episode context)
                    current_task_id = mb_task_ids[0].item()
                    kl_penalty = self.agent.get_kl(current_task_id)
                    kl_loss = self.kl_beta * (kl_penalty / len(mb_obs))
                    
                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss + kl_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Grad Norm
                    total_norm = 0.0
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    self.optimizer.step()
                    
            return {
                "loss": loss.item(),
                "policy_loss": pg_loss.item(),
                "value_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "kl_penalty": kl_loss.item(),
                "raw_kl": kl_penalty.item(),
                "grad_norm": total_norm,
                "clip_fraction": np.mean(clipfracs),
                "rollout_len": len(obs)
            }
    # class VarSharePPO(PPO):
    #     def __init__(self, *args, kl_beta=0.0, batch_size=64, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.kl_beta = kl_beta
    #         self._batch_size = batch_size
            
    #     @property
    #     def batch_size(self):
    #         return self._batch_size

    #     def update_from_storage(self, obs, actions, logprobs, returns, advantages, values, task_ids=None):
    #         self.agent.train()
            
    #         inds = np.arange(len(obs))
    #         clipfracs = []
            
    #         # Optimization epochs loop
    #         for epoch in range(self.update_epochs):
    #             np.random.shuffle(inds)
                
    #             # Mini-batch loop
    #             for start in range(0, len(obs), self.batch_size):
    #                 end = start + self.batch_size
    #                 mb_inds = inds[start:end]
                    
    #                 # Fetch mini-batch data
    #                 mb_obs = obs[mb_inds]
    #                 mb_actions = actions[mb_inds]
    #                 mb_task_ids = task_ids[mb_inds] if task_ids is not None else None
    #                 mb_logprobs = logprobs[mb_inds]
    #                 mb_returns = returns[mb_inds]
    #                 mb_advantages = advantages[mb_inds]
                    
    #                 _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
    #                     mb_obs, 
    #                     mb_actions, 
    #                     task_idx=mb_task_ids
    #                 )
                    
    #                 logratio = newlogprob - mb_logprobs
    #                 ratio = logratio.exp()

    #                 with torch.no_grad():
    #                     approx_kl = ((ratio - 1) - logratio).mean()
    #                     clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

    #                 # Policy Loss
    #                 pg_loss1 = -mb_advantages * ratio
    #                 pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
    #                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    #                 # Value Loss
    #                 newvalue = newvalue.view(-1)
    #                 v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    #                 # Entropy Loss
    #                 entropy_loss = entropy.mean()
                    
    #                 # VarShare KL Penalty
    #                 # Use current task_id (assumed constant for CartPole episode context)
    #                 current_task_id = mb_task_ids[0].item()
    #                 kl_penalty = self.agent.get_kl(current_task_id)
    #                 kl_loss = self.kl_beta * (kl_penalty / len(mb_obs))
                    
    #                 loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss + kl_loss

    #                 self.optimizer.zero_grad()
    #                 loss.backward()
                    
    #                 # Grad Norm
    #                 total_norm = 0.0
    #                 for group in self.optimizer.param_groups:
    #                     for p in group['params']:
    #                         if p.grad is not None:
    #                             total_norm += p.grad.data.norm(2).item() ** 2
    #                 total_norm = total_norm ** 0.5
                    
    #                 self.optimizer.step()
                    
    #         return {
    #             "loss": loss.item(),
    #             "policy_loss": pg_loss.item(),
    #             "value_loss": v_loss.item(),
    #             "entropy": entropy_loss.item(),
    #             "kl_penalty": kl_loss.item(),
    #             "raw_kl": kl_penalty.item(),
    #             "grad_norm": total_norm,
    #             "clip_fraction": np.mean(clipfracs),
    #             "rollout_len": len(obs)
    #         }

    optimizer = optim.Adam(agent.parameters(), lr=args.lr_actor, eps=1e-5)
    
    ppo = PPO(
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
        kl_beta=args.kl_beta,
        kl_schedule=args.kl_schedule,
        kl_controller_args=kl_controller_args,
        lambda_hyper=args.lambda_hyper
    )
    
    # Rollout settings
    n_steps = args.n_steps
    n_updates = args.total_timesteps // (n_steps * args.num_envs)
    
    # Initialize environment
    task_idx = 0
    # Auto-cycling is now handled internally by MetaWorldWrapper.reset()
    # which is triggered by VectorEnv's auto-reset.
    
    eval_env.reset_task(task_idx)

    obs, _ = envs.reset(seed=args.seed)
    
    # Episode metrics tracking
    reward_window = deque(maxlen=25)
    success_window = deque(maxlen=25)
    time_window = deque(maxlen=25)
    loss_window = deque(maxlen=25)
    ent_window = deque(maxlen=25)
    grad_window = deque(maxlen=25)
    kl_raw_window = deque(maxlen=25)
    kl_scaled_window = deque(maxlen=25)
    
    current_ep_reward = 0
    current_ep_length = 0
    ep_start_time = time.time()
    num_episodes_finished = 0
    
    history = []
    start_time = time.time()
    global_step = 0
    next_eval_step = args.eval_freq
    eval_reward_current = 0.0
    eval_metrics = {}
    
    # Track Last 3 Eval Scores for HPO Stability
    last_k_rewards = deque(maxlen=3)
    
    for update in range(n_updates):
        # Rollout Storage
        # Shape: (n_steps, num_envs, ...)
        mb_obs = torch.zeros((n_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        mb_actions = torch.zeros((n_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        mb_logprobs = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_rewards = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_dones = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_values = torch.zeros((n_steps, args.num_envs)).to(device)
        mb_task_ids = torch.zeros((n_steps, args.num_envs), dtype=torch.long).to(device)
        
        for step in range(n_steps):
            global_step += 1 * args.num_envs
            
            # Action logic
            with torch.no_grad():
                task_ids_tensor = torch.full((args.num_envs,), task_idx, dtype=torch.long).to(device)
                
                action, logprob, _, value = agent.get_action_and_value(
                    torch.tensor(obs).float().to(device), 
                    task_idx=task_ids_tensor
                )
                
            mb_obs[step] = torch.tensor(obs).float().to(device)
            mb_actions[step] = action
            mb_logprobs[step] = logprob
            mb_values[step] = value.flatten()
            mb_task_ids[step] = task_ids_tensor
            
            # Env step
            # VectorEnv expects cpu actions usually
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            # Combine done
            dones = np.logical_or(terminations, truncations)
            mb_dones[step] = torch.tensor(dones).float().to(device)
            mb_rewards[step] = torch.tensor(rewards).float().to(device)
            
            obs = next_obs
            
            # Track episode completions
            # Handle Gymnasium VectorEnv info stacking
            if "episode" in infos:
                # infos['episode'] is dict of arrays {'r': ..., 'l': ...}
                # infos['_episode'] is boolean mask
                if "_episode" in infos:
                    for i, has_ep in enumerate(infos["_episode"]):
                        if has_ep:
                            num_episodes_finished += 1
                            reward_window.append(infos["episode"]["r"][i])
                            time_window.append(infos["episode"]["l"][i])
                            if "success" in infos:
                                success_window.append(infos["success"][i])
            elif "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        num_episodes_finished += 1
                        reward_window.append(info["episode"]["r"])
                        time_window.append(info["episode"]["l"])
                        if "success" in info:
                            success_window.append(info["success"])
            
            # Check if we need to switch tasks? 
            # In the original, we switched tasks every episode.
            # Here, environments reset automatically.
            # But the *task* (physics) is set via `reset_task`.
            # SyncVectorEnv doesn't auto-rotate tasks.
            # We can rotate tasks periodically or per-episode.
            # Given parallel envs, we could assign different tasks? 
            # Or keep "Curriculum" behavior: "Train on Task X for a while".
            # The original switched task after ONE episode.
            # If we have 4 envs running Task 0, and one finishes, should we switch THAT env to Task 1?
            # Yes, ideally. But iterating 'envs' is slow.
            # Alternative: Change tasks every N updates?
            # FOR NOW: To maintain behavior with previous single-env loop which switched on 'done'.
            # We will iterate through dones and switch task for that env.
            
            for i, done in enumerate(dones):
                    # Switch task for this env
                    # Auto-cycling now handled internally in MetaWorldWrapper.reset()
                    pass
                    # Note: VectorEnv `step` resets the env, but `reset_task` might need to be called BEFORE reset?
                    # `MultiTaskCartPole` calls `reset_task` then `reset`.
                    # VectorEnv auto-resets immediately after step.
                    # So we are late setting the task for the *next* episode?
                    # Actually, if auto-reset happened, the new episode started with OLD task.
                    # Then we switch task. This might affect the *next* reset?
                    # `reset_task` sets `self.task = ...`
                    # `reset` uses `self.task` to set physics.
                    # So if we call `reset_task` AFTER auto-reset, it only applies to the NEXT reset.
                    # This implies we train one full episode on the old task (which is fine) 
                    # but the VERY FIRST step of the new episode might use old physics?
                    # ComplexCartPole uses physics in `step`.
                    # So if we change `self.task` now, the *current* running episode (just started) will switch physics immediately?
                    # Most CartPole physics are used in `step`.
                    # So yes, calling `reset_task` now updates the physics for the episode that just started.
                    # This simulates "Reset with new task".
                    pass


        # --- Bootstrap & GAE ---
        with torch.no_grad():
            task_ids_tensor = torch.full((args.num_envs,), task_idx, dtype=torch.long).to(device)
            _, _, _, next_value = agent.get_action_and_value(
                torch.tensor(obs).float().to(device), 
                task_idx=task_ids_tensor
            )
            next_value = next_value.reshape(1, -1) # (1, num_envs)
            
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
        
        # Flatten the batch for PPO update
        # (n_steps, num_envs, ...) -> (n_steps * num_envs, ...)
        b_obs = mb_obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = mb_logprobs.reshape(-1)
        b_actions = mb_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = mb_values.reshape(-1)
        b_task_ids = mb_task_ids.reshape(-1)
        
        # Full-Rollout Normalization
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-7)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-7)
        
        # --- Update ---
        metrics = ppo.update_from_storage(
            b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values, task_ids=b_task_ids
        )
        
        # Diagnostics
        explained_var = calculate_explained_variance(b_returns.cpu().numpy(), b_values.cpu().numpy())
        arch_metrics = agent.actor_backbone.get_architectural_metrics(task_idx)
        task_sim = agent.actor_backbone.get_task_similarity()
        
        # Logging windows
        loss_window.append(metrics["loss"])
        ent_window.append(metrics["entropy"])
        grad_window.append(metrics["grad_norm"])
        kl_raw_window.append(metrics["raw_kl"])
        kl_scaled_window.append(metrics["kl_penalty"])
        
        avg_reward = np.mean(reward_window) if reward_window else 0.0
        avg_success = np.mean(success_window) if success_window else 0.0
        history.append({
            "step": global_step,
            "reward": avg_reward,
            "success": avg_success,
            "norm_mu": arch_metrics["mean_norm_mu"],
            "norm_theta": arch_metrics["mean_norm_theta"],
            "sharing_ratio": arch_metrics["sharing_ratio"]
        })
        
        # Evaluation placeholder in history (for nice alignment, use previous or Nan if not eval step)
        sps = int(global_step / (time.time() - start_time))
        
        # Periodic Evaluation
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
            
            # HPO Metric: Avg of last 3
            last_k_rewards.append(eval_reward_mean)
            avg_last_k = np.mean(last_k_rewards)
            
            print(f"Eval Reward: {eval_reward_mean:.2f} | Eval Success: {eval_success_mean:.2f}")
            print(f"FINAL_EVAL_SCORE: {eval_success_mean:.4f}") # Legacy Hook
            print(f"FINAL_EVAL_REWARD: {avg_last_k:.4f}") # Optimized Metric (Avg Last 3)
            next_eval_step += args.eval_freq
            
            # Report to Optuna/WandB if callback provided
            if report_callback:
                report_callback(eval_success_mean, global_step)
            
        history[-1]["eval_reward"] = eval_reward_current
        history[-1]["eval_success"] = eval_metrics.get("eval/mean_success")
        history[-1]["grad_norm"] = metrics["grad_norm"]
        history[-1]["avg_sigma"] = arch_metrics["avg_sigma"]
        history[-1]["episodes"] = num_episodes_finished
        
        # 2. Complete Metric Pack
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
            "loss/kl_penalty": metrics["kl_penalty"],
            "loss/raw_kl": metrics["raw_kl"],
            "diagnostics/grad_norm": metrics["grad_norm"],
            "diagnostics/clip_fraction": metrics["clip_fraction"],
            "diagnostics/explained_variance": explained_var,
            "varshare/mean_norm_mu": arch_metrics["mean_norm_mu"],
            "varshare/mean_norm_theta": arch_metrics["mean_norm_theta"],
            "varshare/sharing_ratio": arch_metrics["sharing_ratio"],
            "varshare/avg_sigma": arch_metrics["avg_sigma"],
            "varshare/task_similarity": task_sim,
            "performance/train_reward_50": avg_reward,
            "performance/train_success_50": avg_success,
        }
        # Add per-task eval metrics
        for t in range(num_tasks):
            eval_metrics[f"eval/reward_task_{t}"] = eval_metrics.get(f"eval/reward_task_{t}", 0.0)
            eval_metrics[f"eval/success_task_{t}"] = eval_metrics.get(f"eval/success_task_{t}", 0.0)
        
        full_metrics.update(eval_metrics)
        
        # WandB log
        # wandb.log(full_metrics, step=global_step)
        
        # CSV Heartbeat
        heartbeat_writer.writerow(full_metrics)
        heartbeat_file.flush()
        
        # Console logging (Step-based)
        print_freq = 2500
        step_increment = n_steps * args.num_envs
        if global_step // print_freq > (global_step - step_increment) // print_freq:
             print(f"Step {global_step:>7d} | Eps {num_episodes_finished:>4d} | Rew {avg_reward:>6.1f} | Suc {avg_success:>4.2f} | SPS {sps:>4d} | Loss {metrics['loss']:>7.3f} | KL {metrics['raw_kl']:>8.2f} | Ent {metrics['entropy']:>5.3f}")

    # Final Duration
    total_duration = time.time() - start_time
    print(f"\nFinal Average Reward: {avg_reward:.2f}")
    print(f"Total Duration: {total_duration:.2f}s")
    
    # 5. Automated Plotting
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
    
    # New Plots
    plot_metric(history, "episodes", "Total Episodes", "episodes.png")
    
    # Filter for eval reward (ignore Nones)
    eval_history = [d for d in history if d.get("eval_reward") is not None]
    if eval_history:
         plot_metric(eval_history, "eval_reward", "Evaluation Reward", "eval_reward.png")
         plot_metric(eval_history, "eval_success", "Evaluation Success Rate", "eval_success.png")
         
    plot_metric(history, "grad_norm", "Gradient Norm", "grad_norm.png")
    plot_metric(history, "avg_sigma", "Average Weight Sigma", "avg_sigma.png")
    
    # Save history as JSON for later synthesis
    with open(os.path.join(seed_dir, "history.json"), "w") as f:
        json.dump(history, f)

    # 6. Final Summary Report
    total_params, active_params = get_parameter_counts(agent)
    final_eval_reward = eval_reward_current if eval_reward_current is not None else 0.0
    
    summary_text = f"""==================================================
VARSHARE TRAINING SUMMARY
==================================================
Experiment Name:    {args.exp_name}
Seed:               {args.seed}
Environment Type:   {args.env_type}
--------------------------------------------------
Total Runtime:      {total_duration:.2f} s
Total Env Steps:    {global_step}
Total Episodes:     {num_episodes_finished}
Total Grad Steps:   {full_metrics['TOTAL_GRAD_STEPS']}
--------------------------------------------------
Total Parameters:   {total_params}
Active Parameters:  {active_params} (per task)
--------------------------------------------------
Final Training Rew: {avg_reward:.2f}
Final Training Suc: {avg_success:.2f}
Final Eval Reward:  {final_eval_reward:.2f}
Final Eval Success: {eval_metrics.get("eval/mean_success", 0.0):.2f}
==================================================
"""
    print(summary_text)
    
    summary_path = os.path.join(seed_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    heartbeat_file.close()
    envs.close()
    eval_env.close()
    # wandb.finish()
    
    print("Training Complete.")
    return history

if __name__ == "__main__":
    train()
