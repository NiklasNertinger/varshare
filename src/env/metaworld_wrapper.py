import gymnasium as gym
import numpy as np
import metaworld
import random

class MetaWorldWrapper(gym.Env):
    """
    Wrapper for Meta-World tasks to make them Gymnasium-compatible
    and support multi-task cycling.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, benchmark="MT10", seed=None, initial_task_idx=0, auto_cycle_task=False):
        super().__init__()
        self.benchmark_name = benchmark
        self.auto_cycle_task = auto_cycle_task
        
        if benchmark == "MT10":
            self.benchmark = metaworld.MT10(seed=seed)
        elif benchmark == "MT50":
            self.benchmark = metaworld.MT50(seed=seed)
        elif benchmark == "MT3":
            # MT3 isn't a native class in all versions, often just a subset
            # We'll pick 3 tasks manually if needed or use MT10 and subselect
            self.benchmark = metaworld.MT10(seed=seed) # Fallback to MT10
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        # Get all tasks
        self.all_tasks = self.benchmark.train_tasks
        self.train_classes = self.benchmark.train_classes
        
        # Filter for MT3 if requested
        if benchmark == "MT3":
            unique_task_names = list(self.train_classes.keys())[:3]
            self.all_tasks = [t for t in self.all_tasks if t.env_name in unique_task_names]
            self.train_classes = {k: v for k, v in self.train_classes.items() if k in unique_task_names}

        self.task_names = list(self.train_classes.keys())
        self.num_tasks = len(self.task_names)
        
        # Current active environment
        self.current_task_idx = initial_task_idx
        self._env = None
        self._set_task_env(initial_task_idx)

        # Observation and Action Spaces
        # Meta-World usually has 39D observations
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _set_task_env(self, task_idx):
        self.current_task_idx = task_idx
        task_name = self.task_names[task_idx]
        
        # Instantiate environment class if first time or different
        if self._env is None or type(self._env) != self.train_classes[task_name]:
            if self._env is not None:
                self._env.close()
            self._env = self.train_classes[task_name]()
        
        # Pick one valid task instance for this task name
        # In newer Meta-World versions, we should be careful with task objects
        valid_tasks = [t for t in self.all_tasks if t.env_name == task_name]
        if not valid_tasks:
             # Fallback: some task objects might not have env_name matching exactly?
             # But MT10/50 should be fine.
             raise ValueError(f"No tasks found for {task_name}")
             
        task = random.choice(valid_tasks)
        self._env.set_task(task)

    def reset(self, seed=None, options=None):
        if self.auto_cycle_task and self.num_tasks > 1:
            next_idx = (self.current_task_idx + 1) % self.num_tasks
            self.reset_task(next_idx)
            
        if seed is not None:
            self._env.action_space.seed(seed)
        obs, info = self._env.reset(seed=seed, options=options)
        # Success tracking
        info["success"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        # Map success metric if available
        # Some versions return info['success'], others info['is_success']
        if 'success' in info:
            info["success"] = float(info["success"])
        elif 'is_success' in info:
            info["success"] = float(info["is_success"])
        else:
            info["success"] = 0.0
            
        return obs, reward, terminated, truncated, info

    def reset_task(self, task_idx):
        """Allows external control of which task to run"""
        self._set_task_env(task_idx)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()
