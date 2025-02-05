"""This module provides agents for the k-karmed bandit Gymnasium environment."""

from collections import defaultdict

import gymnasium as gym
import numpy as np


class BanditAgents:
    def __init__(self):
        self.agents = ["random", "greedy", "epsilon-greedy"]


class BanditAgent(BanditAgents):
    def __init__(self, env: gym.Env, policy: str):
        super().__init__()
        self.env = env
        assert policy in self.agents
        self.policy = policy

        self.training_error = []

    def get_action(self) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if self.policy == "random":
            return self.env.action_space.sample()
        else:
            pass
    
    def update(self, reward):
        optimal_reward = max(self.env.get_wrapper_attr("arms_true_q_values"))
        self.training_error.append(optimal_reward - reward)
