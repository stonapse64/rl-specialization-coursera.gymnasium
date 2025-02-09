"""This module provides agents for the k-karmed bandit Gymnasium environment."""

import random
import gymnasium as gym
import numpy as np
import random

# # TODO replace random.random() with the use of the numpy.generator
# # TODO create a q_table that logs the expected q-value of each arm and the times it has been used. 
# # TODO correct the statement accordingly
# # TODO adapt the function update to update the q-table. We consider the last reward the observation so we want to call self.update before drawing the next action in get_action > else
# # TODO create a function self.reset to reset the agent to its state right after __init__


class BanditAgentEpsilonGreedy:
    def __init__(self, env: gym.Env, epsilon: float = 0.1, start_value: float = 0.0, seed: int = None):
        super().__init__()
        self.env = env
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon
        self.last_reward = 0
        self.start_value = start_value
        self.q_table = np.full(self.env.action_space.n, self.start_value)
        self.visit_count = np.zeros(self.env.action_space.n)
        self.rng = np.random.Generator(np.random.Philox(seed=seed))
        self.max_q_index = np.argmax(self.q_table)
        self.last_action = self.get_action()

    def get_action(self) -> int:
        if self.rng.uniform() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.max_q_index
        self.last_action = action
        return action

    def update(self, reward):
        self.last_reward = reward
        self.visit_count[self.last_action] += 1
        delta = reward - self.q_table[self.last_action]
        self.q_table[self.last_action] += delta / self.visit_count[self.last_action]
        self.max_q_index = np.argmax(self.q_table)

    def reset(self):
        self.q_table = np.full(self.env.action_space.n, self.start_value)
        self.visit_count = np.zeros(self.env.action_space.n)
        self.last_reward = 0
        self.last_action = self.get_action()
        self.max_q_index = np.argmax(self.q_table)