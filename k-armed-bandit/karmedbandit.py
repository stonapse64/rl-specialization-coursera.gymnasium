"""This module provides a k-armed bandit Gymnasium environment."""

import math
import os
import struct
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env


class BanditParams:
    # TODO: write docstring
    '''Parameters for the k-armed BanditEnv environment.'''
    def __init__(
        self,
        bandit_actions: int = 10,
        true_q_value_mean = 0.5, # for a specific initial q-value set this to the value of choice otherwise use mean = 0
        true_q_value_std = 0., # for a uniform initial q-value set this to zero otherwise use e.g., std = 1.
        q_value_std = 1.0, # for q_values drawn from a normal standard distribution set this to 1 else leave it to 0. The true_q_value_mean of each will be used as mean in any case which will be drifting if set so in the next two params.
        qdrift_mean: float = 0.0, # for a stationary problem set this to zero
        qdrift_std: float = 0.0, # for a stationary problem set this to zero else use e.g. std = 0.1 for a light drift
        random_seed: int = 42,

    ):
        self.bandit_actions = bandit_actions
        self.true_q_value_mean = true_q_value_mean
        self.true_q_value_std = true_q_value_std
        self.q_value_std = q_value_std
        self.qdrift_mean = qdrift_mean
        self.qdrift_std = qdrift_std
        self.random_seed = random_seed

        # print("Bandit configuration\n--------------------")
        # print(*[f"{key}: {value}" for key, value in self.__dict__.items()], sep="\n")
    
    def print_bandit_params(self):
        print("Bandit configuration\n--------------------")
        print(*[f"{key}: {value}" for key, value in self.__dict__.items()], sep="\n")


class BanditEnv(gym.Env, BanditParams):
    '''docstring'''

    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        super().__init__()

        # Create an instance of the numpy random generator so that we can seed
        # it for the use in this class without side effects to other parts of 
        # our code or libraries
        self.rng = np.random.default_rng(seed=self.random_seed)

        # observation and action space of the BanditEnv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.bandit_actions,), dtype=np.float64)
        self.action_space = spaces.Discrete(self.bandit_actions)

        # The arrays that holds the true and actual q-values of each bandit arm
        self.arms_true_q_values = np.zeros(self.bandit_actions)
        self.arms_q_values = np.zeros(self.bandit_actions)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        # TODO set correct window_size based on self.arms
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode        
        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # set the true and intial q-values of the bandit.actions bandit arms
        self.arms_true_q_values = self.rng.normal(
            loc=self.true_q_value_mean, 
            scale=self.true_q_value_std, 
            size=self.bandit_actions)
        
        self.arms_q_values = self.arms_true_q_values

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        assert action in self.action_space

        self.arms_q_values = np.array([
            self.rng.normal(loc=true_q, scale=self.q_value_std)
            for true_q in self.arms_true_q_values
        ])

        observation = self._get_obs()
        reward = self.arms_q_values[action]
        terminated = False  # Determine if the episode is terminated
        truncated = False # Determine if the episode is truncated (e.g., due to a timeout)
        info = self._get_info()  # Any additional information

        self._q_drift()

        return observation, reward, terminated, truncated, info
    
    def render(self):
      # ... render the environment (optional)
      pass

    def close(self):
        # ... any cleanup ...
        pass

    def _q_drift(self):
        # TODO: implement true logic
        """
        Updates the bandit with drift for a non-stationary problem.
        """
        # exemplary code from old bandit class
        # if not self.stationary:
        #     increments = np.random.normal(self.q_drift_mean, self.q_drift_std_dev, self.arms.shape)
        #     self.arms += increments

    def _get_obs(self):
        observation = self.arms_q_values
        return observation
    
    def _get_info(self):
        return {}


if __name__ == '__main__':
    # The output in the terminal will indicate whether tests passed or failed.
    env = BanditEnv()
    # XXX info: gymnasium env self test. Run as required. 
    # The output in the terminal will indicate whether tests passed or failed.
    # check_env(env = env)

    print(env.true_q_value_mean)

    obs, info = env.reset()

    print("Initial arms_q_values:", env.arms_true_q_values)
    print ("#"*20)

    for _ in range(10):  # Perform some steps
        action = env.action_space.sample() #Example
        obs, reward, terminated, truncated, info = env.step(action)
        print("Updated arms_q_values:", np.round(env.arms_q_values, 3))

    # env.print_bandit_params()





