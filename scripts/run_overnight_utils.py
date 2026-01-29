
import os
import sys

# Add project root to path so 'src' can be imported when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

from src.env.metaworld_wrapper import MetaWorldWrapper
from src.models import ActorCritic, PaCoActorCritic, SoftModularActorCritic
from src.algo.ppo import PPO
from src.algo.meta_learning import reptile_update

def make_env(seed, idx, env_id="MT10", capture_video=False, run_name=""):
    def thunk():
        if env_id == "MT10":
            env = MetaWorldWrapper(benchmark="MT10", seed=seed, initial_task_idx=idx % 10, auto_cycle_task=True)
        else:
            env = gym.make(env_id)
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def run_seed(
    seed,
    experiment_name,
    algo_config,
    total_timesteps=2500000,
    num_envs=4,
    lr_actor=1e-4,
    lr_critic=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    kl_beta=0.0005,
    max_grad_norm=0.5,
    batch_size=2048, # 4 envs * 512 steps
    minibatch_size=64,
    num_steps=512,  # Increased to maintain batch size with fewer envs
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Setup Paths
    run_dir = f"analysis/overnight_mt10/{experiment_name}/seed_{seed}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Resume Logic
    progress_path = f"{run_dir}/progress.csv"
    if os.path.exists(progress_path):
        print(f"Seed {seed} already completed (found {progress_path}). Skipping...")
        return
    
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Env Setup
    # Windows OOM Protection strategies
    if os.name == 'nt':
        print(f"Windows detected: Limiting num_envs to 2 (was {num_envs}) to prevent MemoryError/BadAllocation.")
        num_envs = min(num_envs, 2)
        # Adjust n_steps to maintain batch size if needed, or just let it be smaller batch (safer)
        # If we reduce envs, we might want to increase n_steps? 
        # But changing n_steps changes the rollout horizon. 
        # Let's keep n_steps fixed and accept smaller batch (more updates per epoch, or just less samples).
        # Actually, PPO batch size = num_envs * n_steps.
        # If we reduce num_envs, we reduce batch size. This is fine for overnight stability.

    envs = gym.vector.AsyncVectorEnv(
        [make_env(seed + i, i, env_id="MT10") for i in range(num_envs)]
    )
    
    # Agent Setup
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    
    # Extract specific config
    use_varshare = algo_config.get("use_varshare", False)
    use_task_embedding = algo_config.get("use_task_embedding", False)
    use_reptile = algo_config.get("use_reptile", False)
    use_reptile = algo_config.get("use_reptile", False)
    hidden_dim = algo_config.get("hidden_dim", 256)
    varshare_args = algo_config.get("varshare_args", {})
    varshare_args = algo_config.get("varshare_args", {})
    
    agent = ActorCritic(
        obs_space,
        act_space,
        hidden_dim=hidden_dim,
        use_task_embedding=use_task_embedding,
        embedding_dim=algo_config.get("embedding_dim", 10),
        num_tasks=10,
        use_varshare=use_varshare,
        varshare_args=varshare_args
    ).to(device)
    
    # Optimizer Split (Actor/Critic)
    if use_varshare:
        # VarShare Model - Explicit Split
        actor_params = list(agent.actor_backbone.parameters()) + list(agent.actor_head.parameters())
        if hasattr(agent, 'actor_logstd'): actor_params += [agent.actor_logstd]
        
        critic_params = list(agent.critic_backbone.parameters()) + list(agent.critic_head.parameters())
        
        optimizer = optim.Adam([
            {'params': actor_params, 'lr': lr_actor},
            {'params': critic_params, 'lr': lr_critic}
        ], eps=1e-5)
    else:
        # Standard/Baseline Model - Heuristic Split
        # Try to find actor/critic modules by name or attribute
        if hasattr(agent, 'actor_mean'):
            # Standard MLP
            actor_params = list(agent.actor_mean.parameters())
            if hasattr(agent, 'actor_logstd'): actor_params += [agent.actor_logstd]
            critic_params = list(agent.critic.parameters())
            
            # Task Embeddings go to both? Or usually just treated as shared/actor?
            # In baselines, usually single LR or shared. Let's put embedding in actor group for now or separate?
            # User output implies actor/critic split.
            # Let's check `train_baseline_ppo.py`. Usually one optimizer for baselines?
            # But user asked for specific LRs.
            # If embedding exists, let's add it to actor group (it inputs to actor).
            if hasattr(agent, 'task_embedding'):
                actor_params += list(agent.task_embedding.parameters())
                
            optimizer = optim.Adam([
                {'params': actor_params, 'lr': lr_actor},
                {'params': critic_params, 'lr': lr_critic}
            ], eps=1e-5)
        else:
             # Fallback single LR if structure unknown (e.g. PaCo might differ)
             optimizer = optim.Adam(agent.parameters(), lr=lr_actor, eps=1e-5)

    # Storage
    obs = torch.zeros((num_steps, num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    # Eval Env (Standard Sync Env for manual iteration)
    # Re-seed with fixed + 100
    eval_env = MetaWorldWrapper(benchmark="MT10", seed=seed+100, initial_task_idx=0, auto_cycle_task=False)
    
    # Store Task IDs for correct replay
    task_ids = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    
    def evaluate_policy(total_eval_episodes=10):
        # Eval loop: Evaluate on all 10 tasks
        avg_successes = []
        avg_returns = []
        for t_idx in range(10):
            eval_env.reset_task(t_idx)
            obs, _ = eval_env.reset()
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            terminated = False
            truncated = False
            ep_ret = 0.0
            while not (terminated or truncated):
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs, task_idx=t_idx, sample=False) # Deterministic
                next_obs, reward, terminated, truncated, info = eval_env.step(action.cpu().numpy()[0])
                ep_ret += reward
                obs = torch.Tensor(next_obs).to(device).unsqueeze(0)
            avg_returns.append(ep_ret)
            avg_successes.append(info.get("success", 0.0))
        return np.mean(avg_returns), np.mean(avg_successes)

    num_updates = total_timesteps // batch_size
    
    # Tracking
    success_window = [] # Legacy list approach, keeps last 100 eps
    return_window = deque(maxlen=20) # Track last 20 updates for reward smoothing

    # Initial Env State
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    current_task_ids = torch.tensor([(i % 10) for i in range(num_envs)], device=device)

    # --- Resume Logic (Intra-seed) ---
    checkpoint_path = f"{run_dir}/checkpoint.pt"
    partial_csv_path = f"{run_dir}/metrics_partial.csv"
    update_start = 1
    metrics_data = []

    if os.path.exists(checkpoint_path) and os.path.exists(partial_csv_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            agent.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            update_start = checkpoint["update"] + 1
            global_step = checkpoint["global_step"]
            if "current_task_ids" in checkpoint:
                current_task_ids = checkpoint["current_task_ids"].to(device)
            
            # Load partial CSV
            metrics_data = pd.read_csv(partial_csv_path).to_dict("records")
            print(f"Resuming from Update {update_start}, Step {global_step}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")
            update_start = 1
            global_step = 0
            metrics_data = []
    else:
        global_step = 0
        # Initial Eval (only if starting fresh)
        print("Initial Evaluation...")
        eval_ret, eval_succ = evaluate_policy()
        print(f"Initial Eval: Return={eval_ret:.2f}, Success={eval_succ:.2f}")

    start_time = time.time()
    
    for update in range(update_start, num_updates + 1):
        # Annealing
        frac = 1.0 - (update - 1.0) / num_updates
        # Handle separate param groups (Actor/Critic)
        if hasattr(optimizer, 'param_groups'):
            # Group 0: Actor (lr_actor), Group 1: Critic (lr_critic)
            # Or generic iteration if not strictly ordered
            # We know the init LRs were lr_actor and lr_critic
            # But simpler rule: scale current group's initial LR (or just assume group 0=actor, 1=critic if length 2)
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[0]["lr"] = frac * lr_actor
                optimizer.param_groups[1]["lr"] = frac * lr_critic
            else:
                # Single group case
                optimizer.param_groups[0]["lr"] = frac * lr_actor

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done
            task_ids[step] = current_task_ids # Store current task IDs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, task_idx=current_task_ids)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            
            # Check for success and cycle tasks
            if "final_info" in infos: 
                for i, info in enumerate(infos["final_info"]):
                    if info and "success" in info:
                        success_window.append(info["success"])
                        if len(success_window) > 100: success_window.pop(0)

                        if use_reptile and use_varshare:
                            finished_task = current_task_ids[i].item()
                            reptile_update(agent, finished_task, alpha=0.1)
                        
                        # Update Task ID belief (matches Wrapper auto-cycle)
                        current_task_ids[i] = (current_task_ids[i] + 1) % 10
            
        # Bootstrap
        with torch.no_grad():
            next_value = agent.get_value(next_obs, task_idx=current_task_ids).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_task_ids = task_ids.reshape(-1) # Correct flattened task IDs from buffer
        
        # Track smoothed return
        return_window.append(b_returns.mean().item())
        
        # PPO Update
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions[mb_inds], 
                    task_idx=b_task_ids[mb_inds] # Use correct IDs
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if True: # Norm Adv
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                if True: # Clip V
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                # VarShare KL Penalty
                if use_varshare:
                    kl_penalty = 0
                    for t_idx in range(10): # MT10
                         kl_penalty += agent.get_kl(t_idx)
                    
                    # Already defined in args, but to be safe use the passed value
                    # Wait, we don't have kl_beta in args here... passed in config?
                    # Ah, we hardcoded 0.0005 in defaults.
                    # Standard PPO loop.
                    loss += kl_beta * kl_penalty
                
                # Metric tracking for logging
                if "train_stats" not in locals(): 
                    train_stats = {
                        "pg": [], "v": [], "ent": [], "kl": [], "clip": [], 
                        "norm_theta": [], "norm_mu": [], "ratio": []
                    }
                train_stats["pg"].append(pg_loss.item())
                train_stats["v"].append(v_loss.item())
                train_stats["ent"].append(entropy_loss.item())
                train_stats["kl"].append(approx_kl.item())
                train_stats["clip"].append(np.mean(clipfracs))
                
                # Arch Metrics (for current task)
                if use_varshare and agent.use_varshare:
                    # Just sample one task for logging overhead reduction? 
                    # Or average over batch if we had task-specific metrics?
                    # get_architectural_metrics returns a dict
                    # We can pick task 0 for logging stability
                    arch_m = agent.actor_backbone.get_architectural_metrics(0)
                    if arch_m:
                        train_stats["norm_theta"].append(arch_m.get("mean_norm_theta", 0))
                        train_stats["norm_mu"].append(arch_m.get("mean_norm_mu", 0))
                        train_stats["ratio"].append(arch_m.get("sharing_ratio", 0))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # ... (PPO logic standard) ...
        
        # Logging Logic
        if update % 10 == 0 or update == num_updates:
            # SPS Calculation
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            
            # ETA Calculation
            frac = update / num_updates
            if frac > 0:
                rem_seconds = (elapsed / frac) - elapsed
                eta_str = time.strftime('%H:%M:%S', time.gmtime(rem_seconds))
            else:
                eta_str = "?"
            
            avg_success = np.mean(success_window) if success_window else 0.0
            avg_ret = np.mean(return_window) if return_window else 0.0
            cur_lr = optimizer.param_groups[0]["lr"]
            
            # Check if this is an Eval Step
            did_eval = False
            eval_ret, eval_succ = 0.0, 0.0
            if update % 50 == 0 or update == num_updates:
                eval_ret, eval_succ = evaluate_policy()
                did_eval = True
            elif metrics_data:
                # Reuse last valid eval for CSV consistency
                eval_ret, eval_succ = metrics_data[-1]["eval_return"], metrics_data[-1]["eval_success"]
            
            # Aggregate training stats
            t_pg = np.mean(train_stats["pg"]) if train_stats.get("pg") else 0.0
            t_v = np.mean(train_stats["v"]) if train_stats.get("v") else 0.0
            t_kl = np.mean(train_stats["kl"]) if train_stats.get("kl") else 0.0
            t_ent = np.mean(train_stats["ent"]) if train_stats.get("ent") else 0.0
            t_ratio = np.mean(train_stats["ratio"]) if train_stats.get("ratio") else 0.0
            
            # Reset stats
            train_stats = {
                "pg": [], "v": [], "ent": [], "kl": [], "clip": [], 
                "norm_theta": [], "norm_mu": [], "ratio": []
            }
            
            print(f"Step {global_step} (Upd {update}/{num_updates}) | SPS: {sps} | LR: {cur_lr:.2e} | ETA: {eta_str}")
            if did_eval:
                print(f"  > Eval  (10 Tasks): Reward={eval_ret:7.2f} | Success={eval_succ:4.2f}")
            print(f"  > Train (Window)  : Reward={avg_ret:7.2f} | Success={avg_success:4.2f}")
            print(f"  > Loss            : Pi={t_pg:6.3f} | V={t_v:6.3f} | KL={t_kl:6.4f} | Ratio={t_ratio:4.2f}")
            print("-" * 60)
            
            metrics_data.append({
                "step": global_step,
                "return": avg_ret,
                "success": avg_success,
                "eval_return": eval_ret,
                "eval_success": eval_succ,
                "pg_loss": t_pg,
                "v_loss": t_v,
                "entropy": t_ent,
                "kl": t_kl,
                "lr": cur_lr,
                "sharing_ratio": t_ratio,
            })
            
            # --- Checkpointing & Memory Hygiene ---
            if update % 50 == 0:
                torch.save({
                    'update': update,
                    'global_step': global_step,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'current_task_ids': current_task_ids,
                }, checkpoint_path)
                pd.DataFrame(metrics_data).to_csv(partial_csv_path, index=False)
                print(f"Checkpoint saved at update {update}")
            
            if update % 10 == 0:
                torch.cuda.empty_cache()
    # Save Data
    df = pd.DataFrame(metrics_data)
    df.to_csv(f"{run_dir}/progress.csv", index=False)
    
    end_time = time.time()
    duration_str = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    print(f"Seed {seed} Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} (Duration: {duration_str})")
    
    # Plot Individual Seed
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Primary Metrics
    sns.lineplot(data=df, x="step", y="success", label="Train", ax=axes[0,0])
    sns.lineplot(data=df, x="step", y="eval_success", label="Eval", ax=axes[0,0])
    axes[0,0].set_title("Success Rate")
    
    sns.lineplot(data=df, x="step", y="return", label="Train", ax=axes[0,1])
    sns.lineplot(data=df, x="step", y="eval_return", label="Eval", ax=axes[0,1])
    axes[0,1].set_title("Returns")
    
    # Row 1, Col 3: Sharing Ratio
    if "sharing_ratio" in df.columns and df["sharing_ratio"].mean() > 0:
        sns.lineplot(data=df, x="step", y="sharing_ratio", ax=axes[0,2], color='orange')
        axes[0,2].set_title("Sharing Ratio")
        axes[0,2].set_ylim(0, 1.05)
    
    # Row 2: Diagnostics
    sns.lineplot(data=df, x="step", y="kl", ax=axes[1,0], color='green')
    axes[1,0].set_title("KL Divergence")
    
    sns.lineplot(data=df, x="step", y="pg_loss", label="Policy", ax=axes[1,1])
    # sns.lineplot(data=df, x="step", y="v_loss", label="Value", ax=axes[1,1]) # Scales differ too much usually
    axes[1,1].set_title("Policy Loss")
    
    sns.lineplot(data=df, x="step", y="entropy", ax=axes[1,2], color='purple')
    axes[1,2].set_title("Entropy")
    
    for ax in axes.flatten(): ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"{experiment_name} - Seed {seed}")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/metrics_curve.png")
    plt.close()
    
    print(f"Seed {seed} results saved to: {os.path.abspath(run_dir)}")
    envs.close()
    eval_env.close()

def run_overnight_experiment(algo_name, varshare_args={}, use_reptile=False, use_task_embedding=False, hidden_dim=64):
    seeds = [1, 2, 3]
    for seed in seeds:
        print(f"\n=== Running {algo_name}, Seed {seed} ===")
        config = {
            "use_varshare": (algo_name != "embedding"),
            "use_task_embedding": use_task_embedding,
            "use_reptile": use_reptile,
            "hidden_dim": hidden_dim,
            "varshare_args": varshare_args
        }
        run_seed(seed, algo_name, config)
        
        # Cleanup between seeds
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Aggregate
    print(f"\n=== Aggregating {algo_name} ===")
    all_data = []
    for seed in seeds:
        path = f"analysis/overnight_mt10/{algo_name}/seed_{seed}/progress.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["seed"] = seed
            all_data.append(df)
    
    if not all_data:
        print(f"Warning: No data found for {algo_name}")
        return
        
    full_df = pd.concat(all_data)
    full_df.to_csv(f"analysis/overnight_mt10/{algo_name}/aggr_results.csv", index=False)
    
    # Aggregated Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Primary Metrics
    sns.lineplot(data=full_df, x="step", y="success", label="Train", errorbar="sd", ax=axes[0,0])
    sns.lineplot(data=full_df, x="step", y="eval_success", label="Eval", errorbar="sd", ax=axes[0,0])
    axes[0,0].set_title("Success Rate (Aggr)")
    
    sns.lineplot(data=full_df, x="step", y="return", label="Train", errorbar="sd", ax=axes[0,1])
    sns.lineplot(data=full_df, x="step", y="eval_return", label="Eval", errorbar="sd", ax=axes[0,1])
    axes[0,1].set_title("Returns (Aggr)")
    
    if "sharing_ratio" in full_df.columns and full_df["sharing_ratio"].mean() > 0:
        sns.lineplot(data=full_df, x="step", y="sharing_ratio", errorbar="sd", ax=axes[0,2], color='orange')
        axes[0,2].set_title("Sharing Ratio")
        axes[0,2].set_ylim(0, 1.05)
        
    # Row 2
    sns.lineplot(data=full_df, x="step", y="kl", errorbar="sd", ax=axes[1,0], color='green')
    axes[1,0].set_title("KL Divergence")
    
    sns.lineplot(data=full_df, x="step", y="pg_loss", errorbar="sd", ax=axes[1,1])
    axes[1,1].set_title("Policy Loss")
    
    sns.lineplot(data=full_df, x="step", y="entropy", errorbar="sd", ax=axes[1,2], color='purple')
    axes[1,2].set_title("Entropy")
    
    for ax in axes.flatten(): ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"{algo_name} (Aggregated Over 3 Seeds)")
    plt.tight_layout()
    plt.savefig(f"analysis/overnight_mt10/{algo_name}/aggr_metrics.png")
    print(f"Aggregated results for {algo_name} saved to: {os.path.abspath(f'analysis/overnight_mt10/{algo_name}/')}")

