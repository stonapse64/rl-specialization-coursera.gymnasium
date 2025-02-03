"""This module provides a k-armed bandit Gymnasium environment."""

import math
import os
import struct
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BanditParams:
    """Parameters for the k-armed BanditEnv environment."""
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


class BanditEnv(gym.Env):
    def __init__(self, params: BanditParams = BanditParams()):
        super().__init__()
        
        # Assign all attributes from params to the BanditEnv instance
        for key, value in params.__dict__.items():
            setattr(self, key, value)


if __name__ == '__main__':
    # The output in the terminal will indicate whether tests passed or failed.
    # TODO: add the gymnasium env self test

    env_params = BanditParams(arms=6, qdrift_mean=0.5)
    for key, value in env_params.__dict__.items():
        print(f'{key}:{value}')

    env = BanditEnv(env_params)
    print(env.arms)  # Should print 6
    print(env.qdrift_mean)  # Should print 0.5

    env = BanditEnv()
    print(env.arms)  # Should print 10
    print(env.qdrift_mean)  # Should print 0.