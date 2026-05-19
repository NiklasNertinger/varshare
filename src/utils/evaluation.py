"""
Evaluation Utilities

This module isolates the deterministic evaluation logic for Multi-Task RL.
It computes the true macro-averaged performance metrics across tasks,
ensuring strict separation between the stochastic rollout collection and
the deterministic scientific evaluation phase.
"""

import torch
import numpy as np
from typing import Any, Dict


def evaluate_agent(agent, envs, device, num_episodes=5, num_tasks=5, eval_tasks=None):
    agent.eval()
    
    if eval_tasks is None:
        eval_tasks = list(range(num_tasks))
        
    # Track completed episode counts, rewards, and successes for each task idx
    completed_rewards = {i: [] for i in eval_tasks}
    completed_successes = {i: [] for i in eval_tasks}
    
    # Track episodic return and success accumulators
    current_rewards = np.zeros(num_tasks)
    current_successes = np.zeros(num_tasks)
    
    obs, info = envs.reset()
    
    # Continue until all environments have completed exactly num_episodes episodes
    while any(len(completed_rewards[i]) < num_episodes for i in eval_tasks):
        obs_tensor = torch.Tensor(obs).to(device)
        # Vectorized tasks: each environment slot i maps to task index i
        task_ids = torch.arange(num_tasks, device=device)
        
        with torch.no_grad():
            import inspect
            sig = inspect.signature(agent.get_action_and_value)
            
            kwargs = {"task_idx": task_ids}
            if 'deterministic' in sig.parameters:
                kwargs['deterministic'] = True
            elif 'sample' in sig.parameters:
                kwargs['sample'] = False
                
            out = agent.get_action_and_value(obs_tensor, **kwargs)
            actions = out[0]
            
        obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
        
        current_rewards += rewards
        
        # Track success rates in info if present
        if "success" in infos:
            current_successes = np.maximum(current_successes, infos["success"])
        
        # Process completed episodes
        dones = terminations | truncations
        for idx in range(num_tasks):
            if dones[idx]:
                if idx in eval_tasks:
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


def evaluate_shared_backbone_only(
    agent: torch.nn.Module,
    eval_envs: Any,
    num_tasks: int,
    device: torch.device,
    eval_episodes: int = 10
) -> Dict[int, Dict[str, float]]:
    """
    Ablation test: Evaluates the VarShare network using ONLY the shared
    backbone parameters (theta), by artificially setting all task-specific
    adapters (mu, beta, gate, gamma) to zero.
    
    This scientifically proves whether the shared backbone isolated a 
    'generalist' representation.
    """
    print("\n>>> Running Ablation: Shared-Backbone Only Evaluation...")
    
    # 1. Cache the original task-specific parameters
    cached_task_params = {}
    for name, param in agent.named_parameters():
        if any(kw in name for kw in ["mus", "betas", "gates", "gammas", "mus_A", "mus_B"]):
            cached_task_params[name] = param.data.clone()
            
    # 2. Zero-out the task-specific parameters
    with torch.no_grad():
        for name, param in agent.named_parameters():
            if any(kw in name for kw in ["mus", "betas", "gates", "gammas", "mus_A", "mus_B"]):
                if "gammas" in name:
                    param.data.fill_(1.0) # Identity scale
                elif "gates" in name:
                    param.data.fill_(1.0) # Open gate
                else:
                    param.data.zero_()    # Zero shift
                
    # 3. Evaluate the "lobotomized" network
    shared_stats = evaluate_agent(agent, eval_envs, device, eval_episodes, num_tasks)
    
    # 4. Restore the exact original parameters
    with torch.no_grad():
        for name, param in agent.named_parameters():
            if name in cached_task_params:
                param.data.copy_(cached_task_params[name])
                
    return shared_stats
