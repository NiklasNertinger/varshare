"""
Environment Vectorization and Wrapper Utilities

This module handles the instantiation of parallel environments,
task-cycling logic, and dynamic normalization of observations/rewards.
"""

import gymnasium as gym
from typing import Callable, Any
from src.env import ComplexCartPole, IdenticalCartPole, MetaWorldWrapper
from src.env.toy_multitask import MultiTaskLunarLander, IdenticalLunarLander
from src.env.normalization import PerTaskVecNormalize


class TaskAutoCycleWrapper(gym.Wrapper):
    """
    Gym wrapper that automatically cycles the environment's task_idx
    to guarantee uniform data distribution across PPO rollout batches.
    """
    def __init__(self, env: gym.Env, num_tasks: int, initial_task_idx: int, auto_cycle: bool):
        super().__init__(env)
        self.num_tasks = num_tasks
        self.current_task = initial_task_idx
        self.auto_cycle = auto_cycle
        
    def reset(self, **kwargs) -> tuple[Any, dict]:
        # Reset the underlying unwrapped environment to the exact task index
        if hasattr(self.env.unwrapped, 'reset_task'):
            self.env.unwrapped.reset_task(self.current_task)
        obs, info = self.env.reset(**kwargs)
        info["task_idx"] = self.current_task
        return obs, info
        
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["task_idx"] = self.current_task
        
        if terminated or truncated:
            if self.auto_cycle:
                # Cycle to the next task ID upon episode completion
                self.current_task = (self.current_task + 1) % self.num_tasks
                
        return obs, reward, terminated, truncated, info


def make_env(env_type: str, seed: int, idx: int, num_tasks: int, initial_task_idx: int = 0, auto_cycle_task: bool = False, mt_setting: str = "mt10", args: Any = None):
    """
    Creates a single thunk that instantiates the environment.
    Applies the TaskAutoCycleWrapper for automated, sequential task execution.
    """
    def thunk() -> gym.Env:
        if env_type == "metaworld":
            env = MetaWorldWrapper(
                benchmark=mt_setting, 
                seed=seed, 
                initial_task_idx=initial_task_idx, 
                auto_cycle_task=auto_cycle_task
            )
        else:
            # Map string identifiers to actual classes
            env_map = {
                "ComplexCartPole": ComplexCartPole,
                "IdenticalCartPole": IdenticalCartPole,
            }
            if env_type in env_map:
                env = env_map[env_type](task_idx=initial_task_idx)
            elif env_type == "MultiTaskLunarLander":
                env = MultiTaskLunarLander(task_idx=initial_task_idx, continuous=getattr(args, 'continuous_lunar_lander', False))
            elif env_type == "IdenticalLunarLander":
                env = IdenticalLunarLander(task_idx=initial_task_idx, continuous=getattr(args, 'continuous_lunar_lander', False))
            else:
                raise ValueError(f"Unknown env_type: {env_type}")
            
            env = TaskAutoCycleWrapper(env, num_tasks, initial_task_idx, auto_cycle_task)
        
        # Standard Gym logging wrapper
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Seed spaces
        env.action_space.seed(seed)
        if hasattr(env.observation_space, 'seed'):
            env.observation_space.seed(seed)
            
        return env
    return thunk


def make_vectorized_envs(args: Any, num_tasks: int) -> tuple[gym.vector.AsyncVectorEnv, gym.vector.AsyncVectorEnv]:
    """
    Instantiates the fully vectorized training and evaluation environments.
    Applies per-task normalization wrappers to protect against magnitude variance.
    """
    is_oracle = getattr(args, "algo", "") == "oracle"
    oracle_task_id = getattr(args, "task_id", 0)
    
    # Determine if we need continuous lunar lander (for specific baseline algorithms)
    if hasattr(args, "algo") and args.algo in ["paco", "soft_mod", "care", "moore"]:
        args.continuous_lunar_lander = True
    else:
        args.continuous_lunar_lander = False
        
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_type=args.env_type,
                seed=args.seed + i, 
                idx=i, 
                num_tasks=num_tasks,
                initial_task_idx=(oracle_task_id if is_oracle else i % num_tasks),
                auto_cycle_task=(not is_oracle),
                mt_setting=args.mt_setting,
                args=args
            ) 
            for i in range(args.num_envs)
        ]
    )
    # Wrap with active normalization tracking for training
    envs = PerTaskVecNormalize(envs, num_tasks, norm_obs=True, norm_reward=True, training=True)
    
    # Evaluation environments do NOT auto-cycle (we want strict deterministic ordering)
    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_type=args.env_type,
                seed=args.seed + 1000 + i, 
                idx=i, 
                num_tasks=num_tasks,
                initial_task_idx=(oracle_task_id if is_oracle else i % num_tasks),
                auto_cycle_task=False,
                mt_setting=args.mt_setting,
                args=args
            ) 
            for i in range(num_tasks)
        ]
    )
    # Wrap with passive normalization (training=False) to prevent corrupting stats during eval
    eval_envs = PerTaskVecNormalize(eval_envs, num_tasks, norm_obs=True, norm_reward=False, training=False)
    
    # Share the underlying running mean/std trackers
    eval_envs.obs_rms = envs.obs_rms
    
    return envs, eval_envs
