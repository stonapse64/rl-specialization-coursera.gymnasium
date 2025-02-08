"""This module provides agents for the k-karmed bandit Gymnasium environment."""

from collections import defaultdict

import gymnasium as gym
import numpy as np


# TODO create the epsilon greedy agent
class BanditAgentEpsilonGreedy():
    def __init__(self, env: gym.Env, epsilon: float = 0.1):
        super().__init__()
        self.env = env
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            _max_val = max(self.q_table[self.DIM_Q])
            action = np.random.choice([i for i, q in enumerate(self.q_table[self.DIM_Q]) if q == _max_val])

        return action
    
    def update(self, reward):
         self.update_q_table(self.reward)

