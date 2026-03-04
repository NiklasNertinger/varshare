
import numpy as np
import sys

# Parameters mimicking your MT4 HPO setup
NUM_ENVS = 8        # Your current parallel envs
N_STEPS = 256       # Your current rollout length
BATCH_SIZE = 64     # Minibatch size
MT_SETTING = 10     # MT10 (10 distinct tasks)

def simulate_ppo_update():
    print(f"--- Simulating PPO Update Step (MT{MT_SETTING}) ---")
    print(f"Conditions: Envs={NUM_ENVS}, Steps={N_STEPS}, Batch={BATCH_SIZE}")
    
    # 1. Create Buffer with Task IDs (Simulating 8 environments)
    # Tasks are assigned [0, 1, 2, 3, 0, 1, 2, 3] to the 8 envs
    # In a rollout, assuming no task cycling mid-episode within 256 steps 
    # (which is true, task changes on done), the buffer looks like this:
    buffer_task_ids = np.zeros((N_STEPS, NUM_ENVS), dtype=int)
    env_assignment = [i % MT_SETTING for i in range(NUM_ENVS)]
    print(f"Environment Task Assignments: {env_assignment}")
    
    for i in range(NUM_ENVS):
        buffer_task_ids[:, i] = env_assignment[i]
        
    # 2. Replicate PPO Flattening & Shuffling Logic (from train_baseline_ppo.py)
    # line 516: b_task_ids = mb_task_ids.reshape(-1)
    b_task_ids = buffer_task_ids.reshape(-1)
    total_samples = len(b_task_ids)
    print(f"Total Samples in Buffer: {total_samples}")
    
    # line 302: np.random.shuffle(b_inds)
    b_inds = np.arange(total_samples)
    np.random.shuffle(b_inds)
    
    # 3. Iterate Minibatches
    print("\n--- Examining Minibatch Contents ---")
    
    num_minibatches = total_samples // BATCH_SIZE
    task_counts_per_batch = []
    
    for start in range(0, total_samples, BATCH_SIZE):
        end = start + BATCH_SIZE
        mb_inds = b_inds[start:end]
        
        # Extract task IDs using shuffled indices
        mb_tasks = b_task_ids[mb_inds]
        
        # Count unique tasks
        unique, counts = np.unique(mb_tasks, return_counts=True)
        count_dict = dict(zip(unique, counts))
        task_counts_per_batch.append(len(unique))
        
        batch_idx = start // BATCH_SIZE
        print(f"Minibatch {batch_idx+1:>2d}/{num_minibatches}: Unique Tasks={len(unique)} | Distribution: {count_dict}")

    # Summary
    avg_unique = np.mean(task_counts_per_batch)
    print(f"\n--- Result ---")
    print(f"Average Unique Tasks per Minibatch: {avg_unique:.2f} / {MT_SETTING}")
    if avg_unique == MT_SETTING:
        print("VERIFIED: Every single minibatch contained samples from ALL 4 tasks.")
    else:
        print("WARNING: Some minibatches missed tasks.")

if __name__ == "__main__":
    simulate_ppo_update()
