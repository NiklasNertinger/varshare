import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from src.models import VarShareLayer, VarShareLoRALayer, ActorCritic

def test_varshare_layer_consistency():
    print("\n--- Testing VarShareLayer Consistency ---")
    in_dim, out_dim = 10, 5
    layer = VarShareLayer(in_dim, out_dim, prior_scale=1.0)
    layer.add_task(0, mu_init=0.0, rho_init=-2.0) # High variance
    
    x = torch.randn(2, in_dim)
    
    # 1. Normal Mode (Local Reparam) - Should vary
    print("Mode: Local Reparam (Default)")
    y1 = layer(x, task_id=0, sample=True)
    y2 = layer(x, task_id=0, sample=True)
    diff = (y1 - y2).abs().sum().item()
    print(f"Diff between two forward passes (should be > 0): {diff}")
    assert diff > 1e-5, "Local reparam should allow variance!"
    
    # 2. Consistent Mode
    print("Mode: Consistent Noise")
    layer.use_consistent_noise = True
    layer.sample_noise(task_ids=[0])
    
    y3 = layer(x, task_id=0, sample=True)
    y4 = layer(x, task_id=0, sample=True)
    diff_cons = (y3 - y4).abs().sum().item()
    print(f"Diff between two forward passes (should be 0): {diff_cons}")
    assert diff_cons < 1e-7, "Consistent noise should be identical!"
    
    # 3. New Sample
    print("Action: Resample Noise")
    layer.sample_noise(task_ids=[0])
    y5 = layer(x, task_id=0, sample=True)
    diff_new = (y3 - y5).abs().sum().item()
    print(f"Diff after resampling (should be > 0): {diff_new}")
    assert diff_new > 1e-5, "Resampling should change output!"
    
    print(">> VarShareLayer Passed")

def test_lora_layer_consistency():
    print("\n--- Testing VarShareLoRALayer Consistency ---")
    in_dim, out_dim = 10, 5
    layer = VarShareLoRALayer(in_dim, out_dim, rank=2)
    layer.add_task(0, mu_init=0.0, rho_init=-2.0)
    
    x = torch.randn(2, in_dim)
    
    # 1. Normal Mode
    y1 = layer(x, task_id=0, sample=True)
    y2 = layer(x, task_id=0, sample=True)
    diff = (y1 - y2).abs().sum().item()
    print(f"Local Reparam Diff: {diff}")
    assert diff > 1e-5
    
    # 2. Consistent Mode
    layer.use_consistent_noise = True
    layer.sample_noise(task_ids=[0])
    
    y3 = layer(x, task_id=0, sample=True)
    y4 = layer(x, task_id=0, sample=True)
    diff_cons = (y3 - y4).abs().sum().item()
    print(f"Consistent Diff: {diff_cons}")
    assert diff_cons < 1e-7
    
    print(">> VarShareLoRALayer Passed")

def test_actor_critic_consistency():
    print("\n--- Testing ActorCritic Full Chain ---")
    import gymnasium as gym
    # Mock Env
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    act_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    
    agent = ActorCritic(
        obs_space, act_space, 
        hidden_dim=32, num_tasks=2, 
        use_varshare=True, 
        varshare_args={"rho_init": -2.0} # High noise
    )
    
    # Init Weights
    for mod in agent.modules():
        if hasattr(mod, "add_task"):
            mod.add_task(0)
            mod.add_task(1)
            
    x = torch.randn(4, 10)
    task_idx = torch.tensor([0, 0, 1, 1])
    
    # 1. Consistent Mode
    agent.set_consistent_noise(True)
    agent.sample_noise(task_ids=[0, 1])
    
    # Forward 1
    # Note: ActorCritic.get_action_and_value samples actions too!
    # But noise in weights determines the MEAN/STD of the action distribution.
    # If weights are fixed, the OUTPUT DISTRIBUTION (logits/mean) should be fixed.
    # The sampled action might differ if we sample actions (stochastic policy).
    # We should check the logits/mean/value.
    
    print("Checking Values and Action Distributions...")
    with torch.no_grad():
        _, _, _, v1 = agent.get_action_and_value(x, task_idx=task_idx)
        _, _, _, v2 = agent.get_action_and_value(x, task_idx=task_idx)
        
    diff_v = (v1 - v2).abs().sum().item()
    print(f"Value Diff (Fixed Noise): {diff_v}")
    assert diff_v < 1e-7, "Values should be identical if weights are fixed!"
    
    # 2. Optimization Step Simulation
    # Check if gradients flow to mu/rho but noise remains fixed.
    print("Checking Optimization flow...")
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    
    # Run a step
    _, _, _, v_initial = agent.get_action_and_value(x, task_idx=task_idx)
    loss = v_initial.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Forward again (Same Noise, New Weights)
    with torch.no_grad():
        _, _, _, v_after = agent.get_action_and_value(x, task_idx=task_idx)
        
    diff_opt = (v_initial - v_after).abs().sum().item()
    print(f"Value Diff after Optimization (Fixed Noise): {diff_opt}")
    assert diff_opt > 1e-6, "Weights updated, so output should change!"
    
    # Check if noise is still same? 
    # Hard to check externally without peeking.
    # But as long as we didn't call sample_noise(), it's fine.
    
    print(">> ActorCritic Passed")

if __name__ == "__main__":
    test_varshare_layer_consistency()
    test_lora_layer_consistency()
    test_actor_critic_consistency()
    print("\nALL TESTS PASSED")
