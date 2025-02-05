import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

class MyEmptyObsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({})  # Or just spaces.Dict()
        self.action_space = spaces.Discrete(2)  # Example action space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return {}, {}  # Fixed: Standard dictionaries

    def step(self, action):
        # ... environment logic ...
        reward = 0  # Calculate the reward
        terminated = False  # Determine if the episode is terminated
        truncated = False # Determine if the episode is truncated (e.g., due to a timeout)
        info = {}  # Any additional information

        return {}, reward, terminated, truncated, {}  # Fixed: Standard dictionary

    def render(self):
      # ... render the environment (optional)
      pass

    def close(self):
        # ... any cleanup ...
        pass


# Example usage:
env = MyEmptyObsEnv()
observation, info = env.reset()
print(f"Initial observation: {observation}") # Output will be {}

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print(f"Observation after step: {observation}") # Output will be {}


 # using the gymnasium check_env, a set of functions for checking an environment implementation. 
 # Source: https://gymnasium.farama.org/_modules/gymnasium/utils/env_checker/
check_env(env = env)

env.close()