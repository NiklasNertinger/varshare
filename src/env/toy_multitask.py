
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict

class MultiTaskCartPole(gym.Env):
    """
    Multi-Task CartPole environment.
    
    Tasks vary by physical parameters:
    0: Standard
    1: High Gravity
    2: Heavy Cart
    3: Heavy Pole
    4: Low Gravity
    5: Long Pole
    """
    def __init__(self, task_idx: int = 0):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.task_idx = task_idx
        
        # Define tasks: (gravity, masscart, masspole, length)
        # Standard: (9.8, 1.0, 0.1, 0.5)
        self.task_configs = [
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5}, # Task 0: Standard
            {'gravity': 12.0, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5}, # Task 1: High Gravity
            {'gravity': 9.8, 'masscart': 2.0, 'masspole': 0.1, 'length': 0.5}, # Task 2: Heavy Cart
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.5, 'length': 0.5}, # Task 3: Heavy Pole
            {'gravity': 5.0, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5}, # Task 4: Low Gravity (Moon-ish)
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 1.0}, # Task 5: Long Pole
        ]
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_task(task_idx)

    @property
    def num_tasks(self):
        return len(self.task_configs)

    def reset_task(self, task_idx: int):
        self.task_idx = task_idx % len(self.task_configs)
        config = self.task_configs[self.task_idx]
        
        # Access the internal environment to modify physics
        unwrapped = self.env.unwrapped
        unwrapped.gravity = config['gravity']
        unwrapped.masscart = config['masscart']
        unwrapped.masspole = config['masspole']
        unwrapped.length = config['length']
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        # Allow passing task_idx in options for convenience
        if options and 'task_idx' in options:
            self.reset_task(options['task_idx'])
            
        # Ensure physics are strictly enforced on every reset
        self.reset_task(self.task_idx) 
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()



class ComplexCartPole(gym.Env):
    """
    Complex CartPole with 5 specific tasks as requested.
    
    Task 1: Baseline CartPole
    Task 2: Short, Light Pole (Fast Dynamics)
    Task 3: Long, Heavy Pole (High Inertia)
    Task 4: Low Gravity
    Task 5: Tight Constraints (Precision Task)
    """
    def __init__(self, task_idx: int = 0):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.task_idx = task_idx
        
        # Default thresholds
        # x_max = 2.4, theta_max = 12 deg (0.20944 rad)
        self.base_x_threshold = 2.4
        self.base_theta_threshold = 12 * 2 * np.pi / 360
        
        # Task Configurations
        self.task_configs = [
            # Task 1: Baseline
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0, 
             'x_threshold': 2.4, 'theta_threshold_radians': 12 * 2 * np.pi / 360},
            
            # Task 2: Short, Light Pole (Fast Dynamics)
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.05, 'length': 0.25, 'force_mag': 10.0, 
             'x_threshold': 2.4, 'theta_threshold_radians': 12 * 2 * np.pi / 360},
            
            # Task 3: Long, Heavy Pole (High Inertia)
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.2, 'length': 1.0, 'force_mag': 10.0, 
             'x_threshold': 2.4, 'theta_threshold_radians': 12 * 2 * np.pi / 360},
             
            # Task 4: Low Gravity
            {'gravity': 6.0, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0, 
             'x_threshold': 2.4, 'theta_threshold_radians': 12 * 2 * np.pi / 360},
             
            # Task 5: Tight Constraints (Precision Task)
            # x_max = 1.5, theta_max = 8 deg (0.13963 rad)
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0, 
             'x_threshold': 1.5, 'theta_threshold_radians': 8 * 2 * np.pi / 360},
        ]
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_task(task_idx)

    @property
    def num_tasks(self):
        return len(self.task_configs)

    def reset_task(self, task_idx: int):
        self.task_idx = task_idx % self.num_tasks
        config = self.task_configs[self.task_idx]
        
        unwrapped = self.env.unwrapped
        
        unwrapped.gravity = config['gravity']
        unwrapped.masscart = config['masscart']
        unwrapped.masspole = config['masspole']
        unwrapped.length = config['length']
        unwrapped.force_mag = config['force_mag']
        
        # Update derived quantities
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length
        
        # Update Thresholds
        unwrapped.x_threshold = config['x_threshold']
        unwrapped.theta_threshold_radians = config['theta_threshold_radians']

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if options and 'task_idx' in options:
            self.task_idx = options['task_idx']
        self.reset_task(self.task_idx)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()



class IdenticalCartPole(gym.Env):
    """
    Identical Multi-Task CartPole environment.
    5 tasks, all identical to standard CartPole.
    Used for benchmarking "how much does sharing hurt/help when tasks are the same".
    """
    def __init__(self, task_idx: int = 0):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.task_idx = task_idx
        
        # Standard configs for all tasks
        self.task_configs = [
            {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0, 
             'x_threshold': 2.4, 'theta_threshold_radians': 12 * 2 * np.pi / 360}
            for _ in range(5)
        ]
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_task(task_idx)

    @property
    def num_tasks(self):
        return len(self.task_configs)

    def reset_task(self, task_idx: int):
        self.task_idx = task_idx % self.num_tasks
        config = self.task_configs[self.task_idx]
        
        unwrapped = self.env.unwrapped
        unwrapped.gravity = config['gravity']
        unwrapped.masscart = config['masscart']
        unwrapped.masspole = config['masspole']
        unwrapped.length = config['length']
        unwrapped.force_mag = config['force_mag']
        
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length
        
        unwrapped.x_threshold = config['x_threshold']
        unwrapped.theta_threshold_radians = config['theta_threshold_radians']

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if options and 'task_idx' in options:
            self.task_idx = options['task_idx']
        self.reset_task(self.task_idx)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()

class MultiTaskLunarLander(gym.Env):
    """
    Multi-Task LunarLander environment.
    
    Tasks vary by gravity, wind, turbulence, and landing target.
    """
    def __init__(self, task_idx: int = 0):
        super().__init__()
        # Using LunarLander-v3 if available, else v2. Assuming Gymnasium is recent.
        try:
            self.env = gym.make('LunarLander-v3')
        except gym.error.Error:
             self.env = gym.make('LunarLander-v2')
             
        self.task_idx = task_idx
        
        self.task_configs = [
            {'gravity': -10.0, 'wind': 0.0, 'turbulence': 0.0, 'target_x': 0.0}, # Task 0: Standard
            {'gravity': -12.0, 'wind': 0.0, 'turbulence': 0.0, 'target_x': 0.0}, # Task 1: High Gravity
            {'gravity': -5.0,  'wind': 0.0, 'turbulence': 0.0, 'target_x': 0.0}, # Task 2: Low Gravity
            {'gravity': -10.0, 'wind': 5.0, 'turbulence': 1.0, 'target_x': 0.0}, # Task 3: Windy
            {'gravity': -10.0, 'wind': -5.0, 'turbulence': 1.0, 'target_x': 0.0}, # Task 4: Windy Left
            {'gravity': -10.0, 'wind': 0.0, 'turbulence': 0.0, 'target_x': -0.5}, # Task 5: Target Left
            {'gravity': -10.0, 'wind': 0.0, 'turbulence': 0.0, 'target_x': 0.5},  # Task 6: Target Right
        ]
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.current_config = self.task_configs[0]
        
        # State variables for shaping reward calculation
        self.prev_shaping_old = None
        self.prev_shaping_new = None

    @property
    def num_tasks(self):
        return len(self.task_configs)

    def reset_task(self, task_idx: int):
        self.task_idx = task_idx % len(self.task_configs)
        self.current_config = self.task_configs[self.task_idx]
        
        unwrapped = self.env.unwrapped
        
        enable_wind = abs(self.current_config['wind']) > 0 or self.current_config['turbulence'] > 0
        
        # Box2D world is `unwrapped.world`.
        if hasattr(unwrapped, 'world') and unwrapped.world is not None:
             unwrapped.world.gravity = (0, self.current_config['gravity'])
        
        # Set wind properties if supported (v3+)
        if hasattr(unwrapped, 'enable_wind'):
            unwrapped.enable_wind = enable_wind
            unwrapped.wind_power = self.current_config['wind']
            unwrapped.turbulence_power = self.current_config['turbulence']
    
    def _get_shaping(self, obs, tx):
        # Extract state. Gymnasium LunarLander obs:
        # [x, y, vx, vy, angle, v_angle, leg1_contact, leg2_contact]
        x, y, vx, vy, angle, v_angle, l, r = obs
        
        # Calculate distance to target
        dist = np.sqrt((x - tx)**2 + y**2)
        v_dist = np.sqrt(vx**2 + vy**2)
        tilt = abs(angle)
        
        # Heuristic shaping reward similar to original Env
        return -100*dist - 100*v_dist - 100*tilt + 10*(l+r)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if options and 'task_idx' in options:
            self.reset_task(options['task_idx'])
        else:
            self.reset_task(self.task_idx)
            
        obs, info = self.env.reset(seed=seed, options=options)
        
        target_x = self.current_config['target_x']
        if target_x != 0:
            obs_raw = obs.copy()
            self.prev_shaping_old = self._get_shaping(obs_raw, 0.0)
            self.prev_shaping_new = self._get_shaping(obs_raw, target_x)
        else:
            self.prev_shaping_old = None
            self.prev_shaping_new = None
            
        return self._transform_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        target_x = self.current_config['target_x']
        
        if target_x != 0:
            # Recompute reward to encourage going to target_x instead of 0
            shaping_new = self._get_shaping(obs, target_x)
            
            if self.prev_shaping_old is not None:
                # recover the original shaping component implicitly (approximation)
                shaping_current_original = self._get_shaping(obs, 0.0) 
                
                # new_reward = old_reward - diff_original + diff_new
                reward = reward - (shaping_current_original - self.prev_shaping_old) + (shaping_new - self.prev_shaping_new)

            self.prev_shaping_old = self._get_shaping(obs, 0.0)
            self.prev_shaping_new = shaping_new
            
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
         # Shift x observation so agent sees target_x as 0
         # The agent is 'ego-centric' wrt target x
         obs_copy = obs.copy()
         obs_copy[0] -= self.current_config['target_x']
         return obs_copy
         
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()
