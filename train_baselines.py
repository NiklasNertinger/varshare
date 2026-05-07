import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from collections import deque
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
from src.models_baselines import ActorCritic
from src.algo.ppo import PPO
from src.algo.pcgrad import PCGrad
from src.algo.cagrad import CAGrad

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="shared", choices=["shared", "oracle", "independent", "pcgrad", "cagrad", "paco", "soft_mod", "care", "moore"], help="baseline algorithm")
    parser.add_argument("--task-id", type=int, default=0, help="task id for oracle baseline")
    parser.add_argument("--num-experts", type=int, default=4, help="number of experts for paco")
    parser.add_argument("--num-modules", type=int, default=4, help="number of modules for soft_mod")
    parser.add_argument("--exp-name", type=str, default="baseline_ppo", help="experiment name")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="total timesteps")
    parser.add_argument("--n-steps", type=int, default=256, help="rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch size")
    parser.add_argument("--num-envs", type=int, default=1, help="number of parallel envs")
    
    # Hyperparameters
    parser.add_argument("--lr-actor", type=float, default=0.001, help="actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=0.0001, help="critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="gae lambda")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="clip coefficient")
    parser.add_argument("--k-epochs", type=int, default=4, help="update epochs")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max gradient norm")
    parser.add_argument("--cagrad-c", type=float, default=0.4, help="CAGrad neighborhood constant c")
    
    # Deprecated/Alias for backward compatibility if needed, but we'll use actor/critic
    parser.add_argument("--lr", type=float, default=0.0003, help="unified learning rate (unused if lr-actor/critic are set)")
    
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() == 'true'), default=True, help="cuda toggle")
    parser.add_argument("--eval-mode", type=bool, default=True, help="Run periodic evaluations")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Eval every N steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per task during eval")
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Root analysis directory")
    
    # Environment selection
    parser.add_argument("--env-type", type=str, default="ComplexCartPole", choices=["ComplexCartPole", "IdenticalCartPole", "metaworld", "MultiTaskLunarLander", "IdenticalLunarLander"], help="Environment type")
    parser.add_argument("--mt-setting", type=str, default="MT10", choices=["MT1", "MT3", "MT4", "MT10", "MT50", "None"], help="Meta-World setting")
    
    # Method Specific LRs
    parser.add_argument("--lr-weights", type=float, default=0.001, help="PaCo weights learning rate")
    parser.add_argument("--lr-routing", type=float, default=0.001, help="SoftMod routing learning rate")
    
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=400, help="Network hidden dimension")
    parser.add_argument("--wandb-project", type=str, default="varshare-bench-mt10", help="W&B project name")

    args = parser.parse_args()
    return args

def calculate_explained_variance(y_true, y_pred):
    var_y = np.var(y_true)
    if var_y == 0: return 0
    return 1 - np.var(y_true - y_pred) / var_y

def get_parameter_counts(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params, total_params # No differentiation for standard models

def evaluate(agent, envs, device, num_episodes=5, num_tasks=5, algo="shared", oracle_task=0):
    agent.eval()
    
    # Identify which task indices we should evaluate
    eval_tasks = [oracle_task] if algo == "oracle" else list(range(num_tasks))
    
    # Track completed episode counts, rewards, and successes for each task idx
    completed_rewards = {i: [] for i in eval_tasks}
    completed_successes = {i: [] for i in eval_tasks}
    
    # Track episodic return and success accumulators
    current_rewards = np.zeros(num_tasks)
    current_successes = np.zeros(num_tasks)
    
    obs, info = envs.reset()
    
    # Continue until all environments we care about have completed exactly num_episodes episodes
    while any(len(completed_rewards[i]) < num_episodes for i in eval_tasks):
        obs_tensor = torch.Tensor(obs).to(device)
        # Vectorized tasks: each environment slot i maps to task index i
        task_ids = torch.arange(num_tasks, device=device)
        
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs_tensor, task_idx=task_ids, sample=False)
            
        obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
        
        current_rewards += rewards
        
        # Track success rates in info if present
        if "success" in infos:
            current_successes = np.maximum(current_successes, infos["success"])
        
        # Process completed episodes
        dones = terminations | truncations
        for idx in range(num_tasks):
            if dones[idx]:
                if idx in completed_rewards:
                    if len(completed_rewards[idx]) < num_episodes:
                        # Extract success if available in final_info
                        success_val = current_successes[idx]
                        if "final_info" in infos and infos["final_info"] is not None:
                            f_info = infos["final_info"][idx]
                            if f_info is not None and "success" in f_info:
                                success_val = f_info["success"]
                                
                        completed_rewards[idx].append(current_rewards[idx])
                        completed_successes[idx].append(success_val)
                        
                # Reset accumulators for the next episode
                current_rewards[idx] = 0.0
                current_successes[idx] = 0.0
                    
    # Format and average results
    task_stats = {}
    for i in eval_tasks:
        task_stats[i] = {
            "reward": np.mean(completed_rewards[i]) if completed_rewards[i] else 0.0,
            "success": np.mean(completed_successes[i]) if completed_successes[i] else 0.0
        }
        
    agent.train()
    return task_stats

def train(report_callback=None):
    args = parse_args()
    timestamp = int(time.time())
    run_name = f"{args.exp_name}_{args.algo}_s{args.seed}_{timestamp}"
    if args.algo == "oracle":
        run_name += f"_t{args.task_id}"
    
    # 1. Directory Structure
    exp_dir = os.path.join(args.analysis_dir, args.exp_name)
    seed_str = f"seed_{args.seed}"
    if args.algo == "oracle":
        seed_str += f"_task_{args.task_id}"
    seed_dir = os.path.join(exp_dir, seed_str)
    os.makedirs(seed_dir, exist_ok=True)
    
    # W&B Initialization
    use_wandb = False
    try:
        wandb.init(
            project=args.wandb_project,
            entity="niklas-nertinger-university-of-oxford",
            name=run_name,
            config=vars(args),
            group=args.algo,
            tags=[args.algo, args.env_type]
        )
        use_wandb = True
    except Exception as e:
        print(f"W&B Initialization Failed: {e}. Running without W&B.")
    
    heartbeat_path = os.path.join(seed_dir, "heartbeat.csv")
    heartbeat_file = open(heartbeat_path, "w", newline="")
    heartbeat_writer = None
    
    # We'll determine num_tasks based on env_type
    if args.env_type == "metaworld":
        # Temporary wrapper to get num_tasks
        temp_env = MetaWorldWrapper(benchmark=args.mt_setting, seed=args.seed)
        num_tasks = temp_env.num_tasks
        temp_env.close()
    else:
        num_tasks = 5
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Algorithm: {args.algo}")
    if args.algo == "oracle":
        print(f"Training on Task ID: {args.task_id}")
    
    # Env Setup
    def make_env(seed, idx, initial_task_idx=0, auto_cycle_task=False):
        def thunk():
            if args.env_type == "ComplexCartPole":
                env = ComplexCartPole()
            elif args.env_type == "IdenticalCartPole":
                env = IdenticalCartPole()
            elif args.env_type == "MultiTaskLunarLander":
                env = MultiTaskLunarLander(task_idx=initial_task_idx)
            elif args.env_type == "IdenticalLunarLander":
                env = IdenticalLunarLander()
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
                initial_task_idx=(args.task_id if args.algo == "oracle" else i % num_tasks),
                auto_cycle_task=(args.algo in ["shared", "pcgrad", "paco", "soft_mod", "care", "moore", "independent"])
            ) 
            for i in range(args.num_envs)
        ]
    )
    
    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.seed + 1000 + i, 
                i, 
                initial_task_idx=(args.task_id if args.algo == "oracle" else i % num_tasks),
                auto_cycle_task=False
            ) 
            for i in range(num_tasks)
        ]
    )
    
    # Model Setup
    hdims = [args.hidden_dim] * 3
    if args.algo == "paco":
        from src.baselines import PaCoActorCritic
        agent = PaCoActorCritic(
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dims=hdims,
            num_tasks=num_tasks,
            num_experts=args.num_experts
        ).to(device)
        
        actor_params = list(agent.actor_backbone.parameters()) + list(agent.actor_head.parameters()) + [agent.actor_logstd]
        critic_params = list(agent.critic_backbone.parameters()) + list(agent.critic_head.parameters())
        
        # Specific Composition Weights
        weight_params = [agent.task_encoder.embedding.weight]

    elif args.algo == "soft_mod":
        from src.baselines import SoftModularActorCritic
        agent = SoftModularActorCritic(
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dims=hdims,
            num_tasks=num_tasks,
            num_modules=args.num_modules
        ).to(device)
        
        # Base Params
        actor_params = list(agent.actor.parameters()) + list(agent.state_encoder.parameters()) + [agent.actor_logstd]
        critic_params = list(agent.critic.parameters())
        
        routing_params = list(agent.task_encoder.parameters())

    elif args.algo == "care":
        from src.baselines import CAREActorCritic
        agent = CAREActorCritic(
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dims=hdims,
            num_tasks=num_tasks,
            num_experts=args.num_experts
        ).to(device)
        
        actor_params = list(agent.actor.parameters()) + [agent.actor_logstd]
        critic_params = list(agent.critic.parameters())

    elif args.algo == "moore":
        from src.baselines import MOOREActorCritic
        agent = MOOREActorCritic(
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dims=hdims,
            num_tasks=num_tasks,
            num_experts=args.num_experts
        ).to(device)
        
        actor_params = list(agent.actor.parameters()) + [agent.actor_logstd]
        critic_params = list(agent.critic.parameters())
        
    elif args.algo == "independent":
        from src.baselines import IndependentActorCritic
        agent = IndependentActorCritic(
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dims=hdims,
            num_tasks=num_tasks
        ).to(device)
        
        actor_params = list(agent.actors.parameters())
        if hasattr(agent, 'actor_logstds'):
            actor_params += [agent.actor_logstds]
        critic_params = list(agent.critics.parameters())
        
    else:
        agent = ActorCritic(
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dim=args.hidden_dim,
            num_tasks=num_tasks if args.algo in ["shared", "pcgrad"] else 1,
            use_task_embedding=(args.algo in ["shared", "pcgrad"]),
            embedding_dim=10 if args.algo in ["shared", "pcgrad"] else 0,
            use_varshare=False
        ).to(device)
        
        # Split Params
        actor_params = list(agent.actor_mean.parameters())
        if hasattr(agent, 'actor_logstd'):
            actor_params += [agent.actor_logstd]
        if hasattr(agent, 'task_embedding'):
            actor_params += list(agent.task_embedding.parameters())
            
        critic_params = list(agent.critic.parameters())
    
    # Construct Optimizer Param Groups
    optim_groups = [
        {'params': actor_params, 'lr': args.lr_actor},
        {'params': critic_params, 'lr': args.lr_critic}
    ]
    
    # Add PaCo Weights if present
    if 'weight_params' in locals():
        optim_groups.append({'params': weight_params, 'lr': args.lr_weights})
        
    # Add SoftMod Routing if present
    if 'routing_params' in locals():
        optim_groups.append({'params': routing_params, 'lr': args.lr_routing})
    
    optimizer = optim.Adam(optim_groups, eps=1e-5)
    
    class PCGradPPO(PPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pcgrad = PCGrad(self.optimizer)

        def update_from_storage(self, obs, actions, logprobs, returns, advantages, values, task_ids=None):
            b_inds = np.arange(len(obs))
            clipfracs = []
            td_vars = []
            
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(obs), self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_task_ids = task_ids[mb_inds] if task_ids is not None else None
                    
                    # PCGrad needs per-task losses
                    unique_tasks = torch.unique(mb_task_ids)
                    task_losses = []
                    
                    # Shared logs for metrics
                    total_pg_loss = 0
                    total_v_loss = 0
                    total_entropy = 0
                    
                    for t in unique_tasks:
                        task_mask = (mb_task_ids == t)
                        if not task_mask.any(): continue
                        
                        t_obs = obs[mb_inds][task_mask]
                        t_actions = actions[mb_inds][task_mask]
                        t_logprobs = logprobs[mb_inds][task_mask]
                        t_returns = returns[mb_inds][task_mask]
                        t_advantages = advantages[mb_inds][task_mask]
                        
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            t_obs, t_actions, task_idx=t
                        )
                        
                        logratio = newlogprob - t_logprobs
                        ratio = logratio.exp()
                        
                        # Policy loss
                        pg_loss1 = -t_advantages * ratio
                        pg_loss2 = -t_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        # Value loss
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - t_returns) ** 2).mean()
                        
                        entropy_loss = entropy.mean()
                        
                        task_loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                        task_losses.append(task_loss)
                        
                        total_pg_loss += pg_loss.item()
                        total_v_loss += v_loss.item()
                        total_entropy += entropy_loss.item()
                        td_vars.append((t_returns - newvalue).var().item())

                        if t == unique_tasks[0]: # Sample for clipfrac
                             with torch.no_grad():
                                clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    if task_losses:
                        self.pcgrad.zero_grad()
                        self.pcgrad.pc_backward(task_losses)
                        
                        # Get grad_norm for logging
                        total_norm = 0.0
                        for group in self.optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        self.pcgrad.step()
                        
            return {
                "loss": (total_pg_loss + total_v_loss) / len(unique_tasks) if len(unique_tasks) > 0 else 0.0,
                "policy_loss": total_pg_loss / len(unique_tasks) if len(unique_tasks) > 0 else 0.0,
                "value_loss": total_v_loss / len(unique_tasks) if len(unique_tasks) > 0 else 0.0,
                "entropy": total_entropy / len(unique_tasks) if len(unique_tasks) > 0 else 0.0,
                "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
                "td_error_variance": np.mean(td_vars) if td_vars else 0.0,
                "grad_norm": total_norm
            }

    class CAGradPPO(PPO):
        def __init__(self, c_val=0.4, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cagrad = CAGrad(self.optimizer, c=c_val)

        def update_from_storage(self, obs, actions, logprobs, returns, advantages, values, task_ids=None):
            b_inds = np.arange(len(obs))
            clipfracs = []
            td_vars = []
            
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(obs), self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_task_ids = task_ids[mb_inds] if task_ids is not None else None
                    
                    # CAGrad needs per-task losses
                    unique_tasks = torch.unique(mb_task_ids)
                    task_losses = []
                    
                    # Shared logs for metrics
                    total_pg_loss = 0
                    total_v_loss = 0
                    total_entropy = 0
                    
                    for t in unique_tasks:
                        task_mask = (mb_task_ids == t)
                        if not task_mask.any(): continue
                        
                        t_obs = obs[mb_inds][task_mask]
                        t_actions = actions[mb_inds][task_mask]
                        t_logprobs = logprobs[mb_inds][task_mask]
                        t_returns = returns[mb_inds][task_mask]
                        t_advantages = advantages[mb_inds][task_mask]
                        
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            t_obs, t_actions, task_idx=t
                        )
                        
                        logratio = newlogprob - t_logprobs
                        ratio = logratio.exp()
                        
                        # Policy loss
                        pg_loss1 = -t_advantages * ratio
                        pg_loss2 = -t_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        # Value loss
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - t_returns) ** 2).mean()
                        
                        entropy_loss = entropy.mean()
                        
                        task_loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                        task_losses.append(task_loss)
                        
                        total_pg_loss += pg_loss.item()
                        total_v_loss += v_loss.item()
                        total_entropy += entropy_loss.item()
                        td_vars.append((t_returns - newvalue).var().item())

                        if t == unique_tasks[0]: # Sample for clipfrac
                             with torch.no_grad():
                                clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    if task_losses:
                        self.cagrad.zero_grad()
                        self.cagrad.pc_backward(task_losses)
                        
                        # Get grad_norm for logging
                        total_norm = 0.0
                        for group in self.optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        self.cagrad.step()

            return {
                "loss": (total_pg_loss + total_v_loss), # Approximate
                "policy_loss": total_pg_loss / len(unique_tasks),
                "value_loss": total_v_loss / len(unique_tasks),
                "entropy": total_entropy / len(unique_tasks),
                "clipfrac": np.mean(clipfracs),
                "td_error_variance": np.mean(td_vars) if td_vars else 0.0,
                "grad_norm": total_norm
            }

    if args.algo == "pcgrad":
        ppo = PCGradPPO(
            agent=agent,
            optimizer=optimizer,
            lr=args.lr,
            clip_coef=args.eps_clip,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            update_epochs=args.k_epochs,
            minibatch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            device=device
        )
        ppo.norm_adv = False # Handled at rollout level
    elif args.algo == "cagrad":
        ppo = CAGradPPO(
            c_val=args.cagrad_c,
            agent=agent,
            optimizer=optimizer,
            lr=args.lr,
            clip_coef=args.eps_clip,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            update_epochs=args.k_epochs,
            minibatch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            device=device
        )
        ppo.norm_adv = False
    else:
        ppo = PPO(
            agent=agent,
            optimizer=optimizer,
            lr=args.lr,
            clip_coef=args.eps_clip,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            update_epochs=args.k_epochs,
            minibatch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            device=device
        )
        ppo.norm_adv = False # Handled at rollout level
    
    # Rollout settings
    n_steps = args.n_steps
    n_updates = args.total_timesteps // (n_steps * args.num_envs)
    
    # Initialize environment tasks
    current_task_idx = args.task_id if args.algo == "oracle" else 0
    # No manual reset needed now, handled by make_env and auto_cycle

    obs, info = envs.reset(seed=args.seed)
    
    # Track the active task ID for each environment
    current_task_ids = np.array([args.task_id if args.algo == "oracle" else i % num_tasks for i in range(args.num_envs)], dtype=np.int64)
    
    # Metrics
    task_reward_window = {t: deque(maxlen=25) for t in range(num_tasks)}
    task_success_window = {t: deque(maxlen=25) for t in range(num_tasks)}
    num_episodes_finished = 0
    history = []
    start_time = time.time()
    eval_time_accumulated = 0.0
    global_step = 0
    next_eval_step = 0
    total_grad_steps = 0
    eval_reward_current = 0.0
    eval_metrics = {}
    start_update = 0
    avg_reward = 0.0
    avg_success = 0.0
    final_eval_reward = 0.0
    update = 0
    
    checkpoint_path = os.path.join(seed_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_update = checkpoint["update"] + 1
        global_step = checkpoint["global_step"]
        total_grad_steps = checkpoint["total_grad_steps"]
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
                action, logprob, _, value = agent.get_action_and_value(
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
                            if "success" in infos:
                                task_success_window[t].append(infos["success"][i])
            elif "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        num_episodes_finished += 1
                        t = info.get("task_idx", current_task_ids[i])
                        task_reward_window[t].append(info["episode"]["r"])
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
            
            # Auto-cycling is now handled internally by MetaWorldWrapper.reset()
            # which is called automatically by VectorEnv when a rollout sub-env finishes.

        # GAE
        with torch.no_grad():
            task_ids_tensor = torch.tensor(current_task_ids, dtype=torch.long).to(device)
            _, _, _, next_value = agent.get_action_and_value(
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
        
        # Full-Rollout Normalization (Matches VarShare)
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-7)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-7)
        
        # Update
        metrics = ppo.update_from_storage(
            b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values, task_ids=b_task_ids
        )
        total_grad_steps += args.k_epochs * (n_steps * args.num_envs // args.batch_size)
        
        explained_var = calculate_explained_variance(b_returns.cpu().numpy(), b_values.cpu().numpy())
        
        # Architecture Metrics (for PaCo / VarShare / SoftMod)
        # Just grab from task 0 for simplicity if multi-task, or average?
        arch_metrics = {}
        if hasattr(agent, "get_architectural_metrics"):
             # Get stats for Task 0 as a sample
             m = agent.get_architectural_metrics(0)
             if m: arch_metrics = m
             
        # Average per-task train rewards
        for t in range(num_tasks):
            eval_metrics[f"performance/train_reward_task_{t}"] = np.mean(task_reward_window[t]) if task_reward_window[t] else 0.0
            eval_metrics[f"performance/train_success_task_{t}"] = np.mean(task_success_window[t]) if task_success_window[t] else 0.0
        
        # Global averages
        all_train_rewards = [r for d in task_reward_window.values() for r in d]
        all_train_successes = [s for d in task_success_window.values() for s in d]
        avg_reward = np.mean(all_train_rewards) if all_train_rewards else 0.0
        avg_success = np.mean(all_train_successes) if all_train_successes else 0.0
        sps = int(global_step / max(1.0, (time.time() - start_time - eval_time_accumulated)))
        
        # Eval
        if args.eval_mode and global_step >= next_eval_step:
            eval_start = time.time()
            print(f"\n>>> Running Evaluation at Step {global_step}...")
            task_stats = evaluate(agent, eval_envs, device, num_episodes=args.eval_episodes, num_tasks=num_tasks, algo=args.algo, oracle_task=args.task_id)
            
            eval_reward_mean = np.mean([s["reward"] for s in task_stats.values()])
            eval_success_mean = np.mean([s["success"] for s in task_stats.values()])
            
            for k, v in task_stats.items():
                eval_metrics[f"eval/reward_task_{k}"] = v["reward"]
                eval_metrics[f"eval/success_task_{k}"] = v["success"]
                
            eval_metrics["eval/mean_reward"] = eval_reward_mean
            eval_metrics["eval/mean_success"] = eval_success_mean
            eval_reward_current = eval_reward_mean
            print(f"Eval Reward: {eval_reward_mean:.2f} | Eval Success: {eval_success_mean:.2f}\n")
            next_eval_step += args.eval_freq
            
            eval_time_accumulated += (time.time() - eval_start)
            
            # Report to Optuna/WandB if callback provided
            if report_callback:
                report_callback(eval_reward_mean, global_step)
        
        history.append({
            "step": global_step,
            "reward": avg_reward,
            "success": avg_success,
            "eval_reward": eval_reward_current,
            "eval_success": eval_metrics.get("eval/mean_success"),
            "grad_norm": metrics.get("grad_norm", 0.0),
            "episodes": num_episodes_finished,
            "sps": sps,
            **{f"arch_{k}": v for k, v in arch_metrics.items()}
        })
        
        full_metrics = {
            "TOTAL_ENV_STEPS": global_step,
            "EPISODES_COMPLETED": num_episodes_finished,
            "TOTAL_GRAD_STEPS": total_grad_steps,
            "WALL_CLOCK_TIME": time.time() - start_time,
            "WALL_CLOCK_TIME_ADJUSTED": time.time() - start_time - eval_time_accumulated,
            "SPS": sps,
            "SPS_ADJUSTED": int(global_step / max(1.0, (time.time() - start_time - eval_time_accumulated))),
            "loss/total": metrics["loss"],
            "loss/policy": metrics["policy_loss"],
            "loss/value": metrics["value_loss"],
            "loss/entropy": metrics["entropy"],
            "diagnostics/grad_norm": metrics.get("grad_norm", 0.0),
            "diagnostics/clip_fraction": metrics.get("clip_fraction", metrics.get("clipfrac", 0.0)),
            "diagnostics/td_error_variance": metrics.get("td_error_variance", 0.0),
            "diagnostics/explained_variance": explained_var,
            "performance/train_mean_reward": avg_reward,
            "performance/train_mean_success": avg_success,
            "eval/mean_reward": eval_metrics.get("eval/mean_reward", 0.0),
            "eval/mean_success": eval_metrics.get("eval/mean_success", 0.0),
            **{f"arch/{k}": v for k, v in arch_metrics.items()}
        }
        # Add per-task eval metrics and train metrics
        for t in range(num_tasks):
            full_metrics[f"performance/train_reward_task_{t}"] = eval_metrics.get(f"performance/train_reward_task_{t}", 0.0)
            full_metrics[f"performance/train_success_task_{t}"] = eval_metrics.get(f"performance/train_success_task_{t}", 0.0)
            full_metrics[f"eval/reward_task_{t}"] = eval_metrics.get(f"eval/reward_task_{t}", 0.0)
            full_metrics[f"eval/success_task_{t}"] = eval_metrics.get(f"eval/success_task_{t}", 0.0)
            
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
        
        if global_step % 2500 < args.num_envs * n_steps:
             print(f"Step {global_step:>7d} | Eps {num_episodes_finished:>4d} | Rew {avg_reward:>6.1f} | SPS {sps:>4d} | Loss {metrics['loss']:>7.3f}")

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
                "total_grad_steps": total_grad_steps,
                "num_episodes_finished": num_episodes_finished,
                "next_eval_step": next_eval_step,
                "eval_reward_current": eval_reward_current,
                "eval_metrics": eval_metrics,
                "task_reward_window": task_reward_window,
                "task_success_window": task_success_window,
                "current_task_ids": current_task_ids,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Save history incrementally
            with open(os.path.join(seed_dir, "history.json"), "w") as f:
                json.dump(history, f)
                
            print("[CHECKPOINT] Safe shutdown complete. Exiting with code 99.")
            sys.exit(99)

    # Final reports
    total_duration = time.time() - start_time
    print(f"\nFinal Average Reward: {avg_reward:.2f}")
    
    # Plotting
    os.makedirs(seed_dir, exist_ok=True)
    def plot_metric(data, key, ylabel, filename):
        plt.figure(figsize=(10, 6))
        steps = [d["step"] for d in data]
        values = [d[key] for d in data if d.get(key) is not None]
        if not values: return
        plt.plot(steps[:len(values)], values)
        plt.xlabel("Total Environment Steps")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Steps ({args.exp_name})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(seed_dir, filename))
        plt.close()

    plot_metric(history, "reward", "Training Reward", "train_reward.png")
    plot_metric(history, "success", "Training Success Rate", "train_success.png")
    plot_metric(history, "episodes", "Total Episodes", "episodes.png")
    plot_metric(history, "grad_norm", "Gradient Norm", "grad_norm.png")
    plot_metric(history, "sps", "SPS", "sps.png")
    eval_history = [d for d in history if d.get("eval_reward") is not None]
    if eval_history:
        plot_metric(eval_history, "eval_reward", "Evaluation Reward", "eval_reward.png")
        plot_metric(eval_history, "eval_success", "Evaluation Success Rate", "eval_success.png")

    with open(os.path.join(seed_dir, "history.json"), "w") as f:
        json.dump(history, f)

    total_params, _ = get_parameter_counts(agent)
    final_eval_reward = eval_reward_current if eval_reward_current is not None else 0.0
    summary_text = f"""==================================================
BASELINE TRAINING SUMMARY ({args.algo.upper()})
==================================================
Experiment Name:    {args.exp_name}
Seed:               {args.seed}
Environment Type:   {args.env_type}
--------------------------------------------------
Total Runtime:      {total_duration:.2f} s
Total Env Steps:    {global_step}
Total Episodes:     {num_episodes_finished}
Total Grad Steps:   {total_grad_steps}
--------------------------------------------------
Total Parameters:   {total_params}
Active Parameters:  {total_params}
--------------------------------------------------
Final Training Rew: {avg_reward:.2f}
Final Training Suc: {avg_success:.2f}
Final Eval Reward:  {final_eval_reward:.2f}
Final Eval Success: {eval_metrics.get("eval/mean_success", 0.0):.2f}
==================================================
"""
    print(summary_text)
    with open(os.path.join(seed_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    # Save Final Checkpoint
    print(f"\n[CHECKPOINT] Saving final state at update {update} (Step {global_step})...")
    final_checkpoint = {
        "update": update,
        "global_step": global_step,
        "total_grad_steps": total_grad_steps,
        "num_episodes_finished": num_episodes_finished,
        "next_eval_step": next_eval_step,
        "eval_reward_current": eval_reward_current,
        "eval_metrics": eval_metrics,
        "task_reward_window": task_reward_window,
        "task_success_window": task_success_window,
        "current_task_ids": current_task_ids,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(final_checkpoint, os.path.join(seed_dir, "checkpoint.pt"))

    heartbeat_file.close()
    envs.close()
    
    # HPO Parsing Hook
    print(f"FINAL_EVAL_REWARD: {final_eval_reward:.4f}")
    
    print("Training Complete.")
    
    # Generate final plots
    print(f">>> Generating scientific plots in {seed_dir}...")
    try:
         # Basic metric plots
         history_path = os.path.join(seed_dir, "history.json")
         with open(history_path, "w") as f:
            json.dump(history, f)
            
         # Create a simple plotting script or function here
         # For now, relying on aggregate_plots or previous implementation
    except Exception as e:
         print(f"Plotting failed: {e}")

    return history

if __name__ == "__main__":
    train()
