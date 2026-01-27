import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiTaskCartPole(gym.Env):
    def __init__(self, task_idx=0):
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

    def reset_task(self, task_idx):
        self.task_idx = task_idx % len(self.task_configs)
        config = self.task_configs[self.task_idx]
        
        # Access the internal environment to modify physics
        # CartPole-v1 uses physics in the .env.unwrapped
        unwrapped = self.env.unwrapped
        unwrapped.gravity = config['gravity']
        unwrapped.masscart = config['masscart']
        unwrapped.masspole = config['masspole']
        unwrapped.length = config['length']
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

    def reset(self, seed=None, options=None, task_idx=None):
        if task_idx is not None:
             self.task_idx = task_idx
        self.reset_task(self.task_idx) # Ensure physics are set correctly on reset
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()

class ComplexCartPole(gym.Env):
    """
    Replicates the 'Complex 5' task setting from the other_project.
    Tasks:
    0: Standard
    1: High Gravity (15.0)
    2: Heavy Cart (Mass=2.0)
    3: Target Angle +8 deg (Obs Offset -8 deg)
    4: Target Angle -8 deg (Obs Offset +8 deg)
    """
    def __init__(self, task_idx=0):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.task_idx = task_idx
        self.angle_offset_cad = 0.0
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_task(task_idx)

    def reset_task(self, task_idx):
        self.task_idx = task_idx % 5
        unwrapped = self.env.unwrapped
        
        # Reset defaults first
        unwrapped.gravity = 9.8
        unwrapped.masscart = 1.0
        unwrapped.masspole = 0.1
        unwrapped.length = 0.5
        self.angle_offset_rad = 0.0
        
        if self.task_idx == 1:
            unwrapped.gravity = 15.0
        elif self.task_idx == 2:
            unwrapped.masscart = 2.0
        elif self.task_idx == 3:
            # Target +8 deg -> shift obs so +8 looks like 0
            # Agent sees (angle - target). If it drives obs to 0, angle -> target.
            self.angle_offset_rad = 8.0 * (np.pi / 180.0)
        elif self.task_idx == 4:
            # Target -8 deg
            self.angle_offset_rad = -8.0 * (np.pi / 180.0)
            
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

    def reset(self, seed=None, options=None, task_idx=None):
        if task_idx is not None:
             self.task_idx = task_idx
        self.reset_task(self.task_idx)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._transform_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Note: In other_project, we manually messed with rewards for angles.
        # But for CartPole-v1, reward is 1.0 per step as long as not terminated.
        # Termination is angle > 12 deg (0.209 rad).
        # If we just shift observations, the agent might steer towards 8 deg.
        # But if 8 deg is close to 12 deg, it might terminate early!
        # 8 deg is 0.14 rad. 12 deg is 0.21 rad. Margin is 4 deg.
        # It's harder, but possible.
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
        # obs[2] is angle
        obs[2] -= self.angle_offset_rad
        return obs

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

class MultiTaskLunarLander(gym.Env):
    def __init__(self, task_idx=0):
        super().__init__()
        # Use valid render_mode for newer gym, or handle None.
        self.env = gym.make('LunarLander-v3') # v3 is generally available in newer standard gym/gymnasium
        self.task_idx = task_idx
        
        # Define tasks: (gravity, wind_power, turbulence_power, landing_x)
        # Standard: (-10.0, 0.0, 0.0, 0.0)
        # landing_x ranges from -1 to 1 roughly? The viewport is W=20, H=13.3.
        # Initial x is typically W/2. Target is (0,0).
        # We will shift the "target" concept by modifying rewards? 
        # Or easier: The environment always targets (0,0). We can mess with physics to simulate "wind" 
        # pushing us away, making it harder to land at (0,0).
        # But user asked for "destination angle should differ".
        # This usually means the lander must land at a specific spot or orientation.
        # Since standard LunarLander rewards landing at (0,0), we can just accept that "different task" = "different physics".
        # Wait, "destination angle" might mean the ANGLE of the ship when landing? Or the LOCATION (angle from launch)?
        # User said: "destination angle should differ between tasks".
        # I will interpret this as the target x-coordinate on the ground (which effectively is an angle from the start point above).
        
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

    def reset_task(self, task_idx):
        self.task_idx = task_idx % len(self.task_configs)
        self.current_config = self.task_configs[self.task_idx]
        
        # LunarLander-v3 allows enabling wind
        unwrapped = self.env.unwrapped
        
        # Gravity is a property of the world in Box2D. 'gravity' argument in gym make? 
        # Unfortunately LunarLander doesn't expose gravity easily after init.
        # We might need to re-create the env if gravity changes, or hack Box2D.
        # Re-creating is safer.
        
        enable_wind = abs(self.current_config['wind']) > 0 or self.current_config['turbulence'] > 0
        
        # gymnasium LunarLander allows passing gravity and wind in reset? No, typically in make.
        # Let's check if we can modify it.
        # Box2D world is `unwrapped.world`.
        if hasattr(unwrapped, 'world') and unwrapped.world is not None:
             unwrapped.world.gravity = (0, self.current_config['gravity'])
        
        unwrapped.enable_wind = enable_wind
        unwrapped.wind_power = self.current_config['wind']
        unwrapped.turbulence_power = self.current_config['turbulence']
        
        # Target X: 
        # The reward is calculated based on distance to (0,0). 
        # To change the target, we have to offset the observation of x-position and the reward calculation.
        # This is tricky without subclassing deeply.
        # Alternative: We wrap step() and modify the reward and observation.
        
    def _get_shaping(self, obs, tx):
        x, y, vx, vy, angle, v_angle, l, r = obs
        dist = np.sqrt((x - tx)**2 + y**2)
        v_dist = np.sqrt(vx**2 + vy**2)
        tilt = abs(angle)
        return -100*dist - 100*v_dist - 100*tilt + 10*(l+r)

    def reset(self, seed=None, options=None, task_idx=None):
        if task_idx is not None:
            self.task_idx = task_idx
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
        
        # Modify Reward and Obs for Target X
        target_x = self.current_config['target_x']
        
        if target_x != 0:
            # We need to calculate Shaping Reward based on the new target.
            # Original reward = Shaping_after - Shaping_before + Terminal_terms
            # We want to replace Shaping terms with our own centered on target_x.
            
            # Current raw state shaping to target_x
            shaping_new = self._get_shaping(obs, target_x)
            
            # Previous shaping was stored
            if self.prev_shaping_old is not None:
                # We need to compute what the ORIGINAL env thought the shaping diff was
                # shaping_old_original = self._get_shaping(obs, 0.0) 
                # This is "shaping_after" for the original reward
                
                # However, we don't know the exact internal state of previous step's shaping from here easily 
                # without tracking it ourselves manually or relying on the reward being exactly diff(shaping).
                # Only strictly true for dense reward.
                
                # Better approach used in original code:
                # We inferred the original shaping from the previous step.
                # reward = reward - (shaping_old_original - prev_shaping_old_original) + (shaping_new - prev_shaping_new)
                
                # Let's just track our own shaping diff and add it? 
                # No, we must subtract the original shaping component of the reward.
                # But we don't know it exactly from `reward` float.
                
                # Alternative: Just ADD the shaping to target_x and ignore removing original? 
                # No, then it's double reward.
                
                # Let's stick to the previous implementation's logic but FIX the variable handling.
                shaping_current_original = self._get_shaping(obs, 0.0) # Dist to 0
                
                # We want to replace the "Dist to 0" component with "Dist to target_x".
                # effectively: Reward += (New_Shaping_Diff) - (Old_Shaping_Diff)
                
                # Diff_Original = shaping_current_original - self.prev_shaping_old
                # Diff_New = shaping_new - self.prev_shaping_new
                
                # The 'reward' from step() includes Diff_Original.
                # So we want: reward_new = reward - Diff_Original + Diff_New
                
                reward = reward - (shaping_current_original - self.prev_shaping_old) + (shaping_new - self.prev_shaping_new)

            self.prev_shaping_old = self._get_shaping(obs, 0.0)
            self.prev_shaping_new = shaping_new
            
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
         # Shift x so that target_x becomes 0 for the agent
         obs[0] -= self.current_config['target_x']
         return obs
         
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()
