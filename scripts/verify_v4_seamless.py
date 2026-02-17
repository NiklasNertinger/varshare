import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from src.models import ActorCritic

def verify_seamlessness():
    print("==================================================")
    print("SEAMLESS VERIFICATION: CONSISTENT NOISE SAMPLING")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Init Agent
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(20,))
    act_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
    
    agent = ActorCritic(
        obs_space, act_space, 
        hidden_dim=64, num_tasks=5, 
        use_varshare=True, 
        varshare_args={"rho_init": -1.0} # High noise for visibility
    ).to(device)
    
    # Initialize tasks
    for t in range(5):
        agent.sample_noise(task_ids=[t]) # Ensure tasks are initialized internally
    
    agent.set_consistent_noise(True)
    
    # --- 1. Multi-Task Independence ---
    print("\n1. Checking Multi-Task Independence...")
    agent.sample_noise(task_ids=[0, 1])
    
    x = torch.randn(1, 20).to(device)
    
    # Forward Task 0
    with torch.no_grad():
        _, _, _, v0_a = agent.get_action_and_value(x, task_idx=0)
        _, _, _, v0_b = agent.get_action_and_value(x, task_idx=0)
        
    # Forward Task 1
    with torch.no_grad():
        _, _, _, v1_a = agent.get_action_and_value(x, task_idx=1)
        _, _, _, v1_b = agent.get_action_and_value(x, task_idx=1)
        
    assert torch.allclose(v0_a, v0_b), "Task 0 consistency failed"
    assert torch.allclose(v1_a, v1_b), "Task 1 consistency failed"
    # Ensure they are different tasks
    assert not torch.allclose(v0_a, v1_a), "Tasks 0 and 1 should have independent noise and mus"
    
    print("   [PASSED] Task 0 and Task 1 have stable, independent outputs.")

    # --- 2. Cross-Module Consistency (Actor vs Critic) ---
    print("\n2. Checking Cross-Module Consistency (Internal State)...")
    # We want to verify that actor_backbone and critic_backbone both have cached noise.
    # We check a random layer in both.
    actor_layer = agent.actor_backbone.layers[0]
    critic_layer = agent.critic_backbone.layers[0]
    
    assert hasattr(actor_layer, "cached_eps"), "Actor layer missing cached_eps"
    assert "0" in actor_layer.cached_eps, "Task 0 noise missing in Actor"
    assert "0" in critic_layer.cached_eps, "Task 0 noise missing in Critic"
    
    # Verify noise tensors are different between actor and critic (they should be independent samples)
    noise_actor = actor_layer.cached_eps["0"]
    noise_critic = critic_layer.cached_eps["0"]
    assert not torch.allclose(noise_actor, noise_critic), "Actor and Critic should have independent noise samples"
    
    print("   [PASSED] Actor and Critic have unique, cached noise buffers.")

    # --- 3. PPO Update Simulation (Persistence) ---
    print("\n3. Checking Persistence through Mock Optimization...")
    agent.sample_noise(task_ids=[0])
    
    # Get initial output
    with torch.no_grad():
        _, _, _, v_start = agent.get_action_and_value(x, task_idx=0)
    
    # Record noise tensor ID to ensure it's the same object (or same content)
    noise_obj_id = id(actor_layer.cached_eps["0"])
    noise_content = actor_layer.cached_eps["0"].clone()
    
    # Run 5 optimization iterations (simulating 5 mini-batches/epochs)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    for i in range(5):
        _, _, _, v = agent.get_action_and_value(x, task_idx=0)
        loss = v.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check noise persistence
        cur_noise = actor_layer.cached_eps["0"]
        assert torch.allclose(cur_noise, noise_content), f"Noise changed at iteration {i}!"
        
    print("   [PASSED] Noise remained bit-identical through 5 gradient updates.")

    # --- 4. Transition Back to Local Reparam ---
    print("\n4. Checking Seamless Toggle Back to Local Reparam...")
    agent.set_consistent_noise(False)
    
    with torch.no_grad():
        y1 = agent.actor_backbone(x, task_id=0)
        y2 = agent.actor_backbone(x, task_id=0)
        
    assert not torch.allclose(y1, y2), "Local Reparam should vary every forward pass"
    print("   [PASSED] Toggle to Local Reparam is functional.")

    print("\n==================================================")
    print("ALL SEAMLESS VERIFICATION TESTS PASSED")
    print("==================================================")

if __name__ == "__main__":
    verify_seamlessness()
