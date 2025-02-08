"""This module provides a k-armed bandit Gymnasium environment."""

import math
import os
import struct
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import pygame

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
# id is for use with gym.make('id'), entry_point is 'file name:class name env'
register(id='BanditEnv-v0', entry_point='karmedbandit:BanditEnv',)


class BanditParams:
    # TODO: write docstring
    '''Parameters for the k-armed BanditEnv environment.'''
    def __init__(
        self,
        bandit_actions: int = 10,
        true_q_value_mean = 0., # for a specific initial q-value set this to the value of choice otherwise use mean = 0
        true_q_value_std = 0.1, # for a uniform initial q-value set this to zero otherwise use e.g., std = 1.
        q_value_std = 0., # for q_values drawn from a normal standard distribution set this to 1 else leave it to 0. The true_q_value_mean of each will be used as mean in any case which will be drifting if set so in the next two params.
        q_drift_mean: float = 0.0, # for a stationary problem set this to zero
        q_drift_std: float = 0.01, # for a stationary problem set this to zero else use e.g. std = 0.1 for a light drift
        random_seed: int = None,

    ):
        self.bandit_actions = bandit_actions
        self.true_q_value_mean = true_q_value_mean
        self.true_q_value_std = true_q_value_std
        self.q_value_std = q_value_std
        self.q_drift_mean = q_drift_mean
        self.q_drift_std = q_drift_std
        self.random_seed = random_seed

        self.print_bandit_params()
    
    def print_bandit_params(self):
        print("-"*25+"\nBandit configuration\n"+"-"*25)
        print(*[f"{key}: {value}" for key, value in self.__dict__.items()], sep="\n")
        print("-"*25)


class BanditEnv(gym.Env, BanditParams):
    '''docstring'''

    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        super().__init__()

        # Create an instance of the numpy random generator so that we can seed
        # it for the use in this class without side effects to other parts of 
        # our code or libraries
        # self.rng = np.random.default_rng(seed=self.random_seed)

        # observation and action space of the BanditEnv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.bandit_actions,), dtype=np.float64)
        self.action_space = spaces.Discrete(self.bandit_actions)

        # The arrays that holds the true and actual q-values of each bandit arm
        self.arms_true_q_values = np.zeros(self.bandit_actions)
        self.arms_q_values = np.zeros(self.bandit_actions)

        # Variables to store and communicate the agents efficiency
        self.step_error = 0.
        self.optimal_action = False

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
        self.rng = np.random.default_rng(seed=self.random_seed)
        super().reset(seed=seed)

        # set the true and intial q-values of the bandit.actions bandit arms
        self.arms_true_q_values = self.rng.normal(
            loc=self.true_q_value_mean, 
            scale=self.true_q_value_std, 
            size=self.bandit_actions)
        
        self.arms_q_values = self.arms_true_q_values

        # reinit Variables to store and communicate the agents efficiency
        self.step_error = 0.
        self.optimal_action = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        assert action in self.action_space

        # compute step arms_q_values based on bandit's configuration
        self.arms_q_values = np.array([
            self.rng.normal(loc=true_q, scale=self.q_value_std)
            for true_q in self.arms_true_q_values
        ])

        # # compute reward and further KPI
        reward = self.arms_q_values[action]
        self._compute_KPI(action, reward)

        info = self._get_info()  # Any additional information

        self._q_drift()
        observation = self._get_obs()

        return observation, reward, False, False, info
    
    def render(self):
      # TODO implement render logic
      # ... render the environment (optional)
      pass

    def close(self):
        # TODO implement close logic
        # ... any cleanup ...
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _q_drift(self):
        """
        Updates the bandit with drift for a non-stationary problem.
        """
        # exemplary code from old bandit class
        increments = self.rng.normal(loc=self.q_drift_mean, scale=self.q_drift_std, size=self.bandit_actions)
        self.arms_true_q_values += increments
    
    def _compute_KPI(self, action, reward):
        max_val = np.max(self.arms_q_values)
        self.step_error = max_val - reward
        self.optimal_action = True if action in [i for i, q in enumerate(self.arms_q_values) if q == max_val] else False

        # # compute reward and further KPI (this is revised by claude to improve performance over my version but actually runs slower...)
        # reward = self.arms_q_values[action]
        # max_val = np.max(self.arms_q_values)
        # self.step_error = max_val - reward
        # max_indices = np.where(self.arms_q_values == max_val)[0]
        # self.optimal_action = action in max_indices

        return

    def _get_obs(self):
        observation = self.arms_q_values
        return observation
    
    def _get_info(self):
        info = {"step_error": self.step_error, "optimal_action": self.optimal_action}
        return info


if __name__ == '__main__':

    # TODO implement proper registration and gym.make()
    # register(id="BanditEnv-v0", entry_point="k-armed-bandit:BanditEnv",)
    # gym.make("BanditEnv-v0")

    # HACK
    env = BanditEnv()
    
    # XXX info: gymnasium env self test. Run as required.
    # The output in the terminal will indicate whether tests passed or failed.
    # check_env(env = env)

    print(env.true_q_value_mean)
    obs, info = env.reset()
    print("Initial arms_q_values:", env.arms_true_q_values)
    print ("#"*79)
    for _ in range(10_000):  # Perform some steps
        action = env.action_space.sample() #Example
        obs, reward, terminated, truncated, info = env.step(action)
    
    print("Updated arms_q_values:", np.round(env.arms_q_values, 3))

    env.close()





