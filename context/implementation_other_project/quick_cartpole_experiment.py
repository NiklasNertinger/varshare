import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from envs import ComplexCartPole
from models import VarShareNetwork
from ppo import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--save_plot', type=str, default='quick_cartpole_plot.png', help='Path to save plot')
    args = parser.parse_args()

    # Configuration (Using ~Best Params found for CartPole PPO)
    SEED = args.seed
    NUM_EPISODES = args.episodes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Best Params
    LR_ACTOR = 1e-3
    LR_CRITIC = 1e-4
    KL_BETA = 2.5e-4
    BATCH_SIZE = 128
    
    print(f"--- Quick CartPole Experiment (Seed {SEED}) ---")
    print(f"Device: {DEVICE}")
    
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Environment
    # Use ComplexCartPole for Multi-Task Experiment
    env = ComplexCartPole(task_idx=0)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # Model
    # 5 Tasks for Complex CartPole
    policy = VarShareNetwork(input_dim, output_dim, [64, 64], num_tasks=5).to(DEVICE)
    value = VarShareNetwork(input_dim, 1, [64, 64], num_tasks=5).to(DEVICE)
    
    agent = PPO(policy, value, lr=LR_ACTOR, kl_beta=KL_BETA, device=DEVICE)
    
    # Training Loop
    rewards = []
    
    import time
    start_time = time.time()
    
    # Track returns per task
    task_returns = {0:[], 1:[], 2:[], 3:[], 4:[]}
    
    for ep in range(NUM_EPISODES):
        ep_start = time.time()
        
        # Cycle tasks: 0, 1, 2, 3, 4, 0...
        current_task_idx = ep % 5
        env.reset_task(current_task_idx)
        
        obs, _ = env.reset()
        done = False
        score = 0
        
        states, actions, logprobs, rews, dones = [], [], [], [], []
        
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = agent.policy(state_t, current_task_idx) 
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            act_val = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(act_val)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(act_val)
            logprobs.append(log_prob.item())
            rews.append(reward)
            dones.append(done)
            
            obs = next_obs
            score += reward
            
        # Update PPO
        memory = list(zip(states, actions, logprobs, rews, dones))
        metrics = agent.update(memory, current_task_idx, is_varshare=True)
        
        rewards.append(score)
        task_returns[current_task_idx].append(score)
        duration = time.time() - ep_start
        
        # Debug Output (same format as other project for comparison)
        print(f"[DEBUG] Ep {ep}: "
              f"EpLen={len(states)}, "
              f"KL_raw={metrics.get('kl_raw', 0):.2f}, "
              f"KL_norm={metrics.get('kl_penalty', 0):.6f}, "
              f"PolicyLoss={metrics.get('loss_actor', 0):.6f}, "
              f"ValueLoss={metrics.get('loss_critic', 0):.6f}")

        if (ep + 1) % 50 == 0:
            avg_50 = np.mean(rewards[-50:])
            print(f"Ep {ep+1}/{NUM_EPISODES} | Score: {score:.1f} | Avg(50): {avg_50:.1f} | Time/Ep: {duration:.2f}s | SPS: {int(score/duration)}")
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    
    # Moving Average
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), moving_avg, color='blue', label=f'Moving Avg ({window})')
    
    plt.title(f"Quick CartPole PPO (Seed {SEED})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.save_plot)
    print(f"Plot saved to {args.save_plot}")
    
if __name__ == "__main__":
    main()
