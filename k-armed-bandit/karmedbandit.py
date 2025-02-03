"""This module provides a Blackjack functional environment and Gymnasium environment wrapper BlackJackJaxEnv."""

import math
import os
import struct
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

@struct.dataclass
class BanditParams:
    """Parameters for the jax Blackjack environment."""

    natural: bool = False
    sutton_and_barto: bool = True

