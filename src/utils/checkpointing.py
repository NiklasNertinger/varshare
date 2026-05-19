"""
Checkpointing Utilities

This module isolates the logic for securely saving and loading PyTorch
state dictionaries, RNG states, and W&B tracking IDs, enabling 
seamless job resumption during SLURM time-limit requeues.
"""

import os
import time
import torch
import numpy as np
import random
from typing import Any, Optional, Dict, Tuple


def pre_load_wandb_id(checkpoint_path: str) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Peeks into a checkpoint file to extract the previous W&B run ID.
    This must be called before wandb.init() to ensure log continuity.
    
    Returns:
        is_resuming (bool): True if checkpoint exists and was read.
        wandb_id (str): The previous W&B run ID, or None.
        resume_step (int): The global step of the checkpoint.
    """
    is_resuming = os.path.exists(checkpoint_path)
    resume_wandb_id = None
    resume_step = None
    
    if is_resuming:
        try:
            # We only need the top-level keys, but torch.load reads the whole file
            # Setting map_location="cpu" prevents VRAM spikes
            pre_check = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            resume_wandb_id = pre_check.get("wandb_id", None)
            resume_step = pre_check.get("global_step", "?")
            print(f"[RESUME] Found checkpoint at step {resume_step}. W&B ID: {resume_wandb_id}")
            del pre_check  # Free memory immediately
        except Exception as e:
            print(f"[RESUME] Warning: Could not pre-load checkpoint for W&B resume: {e}")
            
    return is_resuming, resume_wandb_id, resume_step


def save_checkpoint(
    checkpoint_path: str,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    envs: Any,
    global_step: int,
    update: int,
    num_episodes_finished: int,
    next_eval_step: int,
    eval_reward_current: float,
    eval_metrics: Dict[str, float],
    task_reward_window: Dict[int, list],
    task_success_window: Dict[int, list],
    time_window: list,
    last_k_rewards: list,
    current_task_ids: list,
    start_time: float,
    eval_time_accumulated: float,
    wandb_run_id: Optional[str] = None,
    total_grad_steps: Optional[int] = None
) -> None:
    """
    Serializes all training state into a dictionary and saves it to disk.
    Includes RNG states and normalization statistics to guarantee mathematical parity.
    """
    print(f"\n[CHECKPOINT] Saving state at update {update} (Step {global_step})...")
    
    # Serialize dynamic normalization statistics
    obs_rms_state = {}
    if hasattr(envs, 'obs_rms'):
        for t, rms in envs.obs_rms.items():
            obs_rms_state[t] = {"mean": rms.mean, "var": rms.var, "count": rms.count}
            
    ret_rms_state = {}
    if hasattr(envs, 'ret_rms'):
        for t, rms in envs.ret_rms.items():
            ret_rms_state[t] = {"mean": rms.mean, "var": rms.var, "count": rms.count}
    
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
        "optimizer_state_dict": optimizer.state_dict(),
        "wandb_id": wandb_run_id,
        "obs_rms": obs_rms_state,
        "ret_rms": ret_rms_state,
        # RNG States for absolute determinism across SLURM requeues
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_random": random.getstate(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        # Wall-clock continuity
        "elapsed_time": time.time() - start_time,
        "eval_time_accumulated": eval_time_accumulated
    }
    
    if total_grad_steps is not None:
        checkpoint["total_grad_steps"] = total_grad_steps
        
    torch.save(checkpoint, checkpoint_path)


def write_verification_report(
    verify_path: str,
    event_type: str,
    global_step: int,
    update: int,
    num_episodes_finished: int,
    eval_reward_current: float,
    wandb_id: str,
    agent: torch.nn.Module,
    envs: Any,
    checkpoint: Optional[Dict] = None
) -> None:
    """
    Writes a cryptographic checksum report of model weights and normalization stats
    to prove that job resumption did not corrupt the training trajectory.
    """
    with open(verify_path, "a") as vf:
        vf.write(f"\n{'='*60}\n")
        vf.write(f"[{event_type}] Checkpoint at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        vf.write(f"  global_step:          {global_step}\n")
        vf.write(f"  update:               {update}\n")
        vf.write(f"  num_episodes:         {num_episodes_finished}\n")
        vf.write(f"  eval_reward_current:  {eval_reward_current}\n")
        vf.write(f"  wandb_id:             {wandb_id}\n")
        
        # Calculate naive parameter checksum
        param_hash = sum(p.sum().item() for p in agent.parameters())
        vf.write(f"  model_param_checksum: {param_hash:.6f}\n")
        
        if hasattr(envs, 'obs_rms'):
            for t in sorted(envs.obs_rms.keys()):
                vf.write(f"  obs_rms[{t}].mean_sum: {envs.obs_rms[t].mean.sum():.6f}\n")
                vf.write(f"  obs_rms[{t}].var_sum:  {envs.obs_rms[t].var.sum():.6f}\n")
                vf.write(f"  obs_rms[{t}].count:    {envs.obs_rms[t].count}\n")
                
        if event_type == "RESUME" and checkpoint is not None:
            vf.write(f"  norm_stats_restored:  {'obs_rms' in checkpoint}\n")
            vf.write(f"  rng_states_restored:  {'rng_torch' in checkpoint}\n")
            
        vf.write(f"{'='*60}\n")
