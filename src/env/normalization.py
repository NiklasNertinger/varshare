import numpy as np
import gymnasium as gym

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

class PerTaskVecNormalize:
    """
    MTBench-style Normalization Wrapper.
    Maintains a separate RunningMeanStd for every task_idx.
    """
    def __init__(self, env, num_tasks, norm_obs=True, norm_reward=True, gamma=0.99, epsilon=1e-8, training=True):
        self.env = env
        self.num_envs = getattr(env, 'num_envs', 1)
        self.observation_space = getattr(env, 'observation_space', None)
        self.action_space = getattr(env, 'action_space', None)
        self.single_observation_space = getattr(env, 'single_observation_space', None)
        self.single_action_space = getattr(env, 'single_action_space', None)
        
        self.num_tasks = num_tasks
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        
        if self.norm_obs:
            self.obs_rms = {t: RunningMeanStd(shape=self.env.single_observation_space.shape) for t in range(num_tasks)}
        
        if self.norm_reward:
            self.ret_rms = {t: RunningMeanStd(shape=()) for t in range(num_tasks)}
            self.ret = np.zeros(self.num_envs)
            
    def step(self, action):
        obs, rews, terms, truncs, infos = self.env.step(action)
        
        task_ids = np.zeros(self.num_envs, dtype=np.int64)
        if "task_idx" in infos:
            task_ids = infos["task_idx"]
            
        if self.norm_obs:
            obs = self._normalize_obs(obs, task_ids)
            
        if self.norm_reward:
            if self.training:
                self.ret = self.ret * self.gamma + rews
                rews = self._normalize_reward(rews, task_ids)
                # Reset returns for done envs
                dones = np.logical_or(terms, truncs)
                self.ret[dones] = 0.0
            else:
                rews = self._normalize_reward(rews, task_ids, update=False)
            
        return obs, rews, terms, truncs, infos
        
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        
        if self.norm_reward:
            self.ret = np.zeros(self.num_envs)
            
        if self.norm_obs:
            task_ids = np.zeros(self.num_envs, dtype=np.int64)
            if "task_idx" in infos:
                task_ids = infos["task_idx"]
            obs = self._normalize_obs(obs, task_ids)
            
        return obs, infos
        
    def _normalize_obs(self, obs, task_ids):
        norm_obs = np.copy(obs)
        for t in range(self.num_tasks):
            mask = task_ids == t
            if np.any(mask):
                if self.training:
                    self.obs_rms[t].update(obs[mask])
                norm_obs[mask] = np.clip(
                    (obs[mask] - self.obs_rms[t].mean) / np.sqrt(self.obs_rms[t].var + self.epsilon),
                    -10.0, 10.0
                )
        return norm_obs
        
    def _normalize_reward(self, rews, task_ids, update=True):
        norm_rews = np.copy(rews)
        for t in range(self.num_tasks):
            mask = task_ids == t
            if np.any(mask):
                if self.training and update:
                    self.ret_rms[t].update(self.ret[mask])
                norm_rews[mask] = np.clip(
                    rews[mask] / np.sqrt(self.ret_rms[t].var + self.epsilon),
                    -10.0, 10.0
                )
        return norm_rews

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
