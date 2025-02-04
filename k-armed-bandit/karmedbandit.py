"""This module provides a k-armed bandit Gymnasium environment."""

import math
import os
import struct
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BanditParams:
    '''Parameters for the k-armed BanditEnv environment.'''
    # TODO add missing params for initial q-values (uniform, randint from range)
    # and draw logic (constant or from a probability distribution with
    #  mean = initial q-values and std)
    def __init__(
        self,
        arms: int = 10,
        qdrift_mean: float = 0.0,
        qdrift_std: float = 0.0,
        stationary: bool = True,
    ):
        self.arms = arms
        self.qdrift_mean = qdrift_mean
        self.qdrift_std = qdrift_std
        self.stationary = stationary

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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


if __name__ == '__main__':
    # The output in the terminal will indicate whether tests passed or failed.
    # TODO: add the gymnasium env self test
    env = BanditEnv()
    env.print_bandit_params()
    env.arms = 6
    env.print_bandit_params()