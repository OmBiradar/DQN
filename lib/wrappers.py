# lib/wrappers.py
from collections import deque
import numpy as np
import gymnasium as gym

class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, fn):
        super().__init__(env)
        self._fn = fn

    def observation(self, observation):
        return self._fn(observation)

class FrameStack(gym.ObservationWrapper):
    """
    Gymnasium-style FrameStack:
    - reset returns (stacked_obs, info)
    - step returns (stacked_obs, reward, terminated, truncated, info)
    """
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = int(num_stack)
        self.frames = deque(maxlen=self.num_stack)

        low = np.repeat(self.observation_space.low[None, ...], self.num_stack, axis=0)
        high = np.repeat(self.observation_space.high[None, ...], self.num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # stack along first dim (N, H, W[, C]) and keep dtype
        return np.stack(list(self.frames), axis=0)
