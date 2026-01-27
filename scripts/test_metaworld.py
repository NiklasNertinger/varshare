import metaworld
import random

def test_metaworld():
    print("Testing Meta-World MT10 setup...")
    
    # 1. Load MT10
    mt10 = metaworld.MT10()
    training_envs = mt10.train_classes
    
    print(f"Number of tasks in MT10: {len(training_envs)}")
    print(f"Task names: {list(training_envs.keys())}")
    
    # 2. Pick a random task
    task_name = random.choice(list(training_envs.keys()))
    print(f"Picking random task: {task_name}")
    
    env_cls = training_envs[task_name]
    env = env_cls()
    
    # 3. Setup task
    task = random.choice([t for t in mt10.train_tasks if t.env_name == task_name])
    env.set_task(task)
    
    # 4. Step
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print(f"Step successful. Reward: {reward}, Done: {done}")
    if 'success' in info:
        print(f"Success metric found: {info['success']}")
    else:
        print("Success metric NOT found in info (Expected in some versions of MW)")

if __name__ == "__main__":
    try:
        test_metaworld()
        print("\nMeta-World verification SUCCESSFUL!")
    except Exception as e:
        print(f"\nMeta-World verification FAILED: {e}")
        import traceback
        traceback.print_exc()
