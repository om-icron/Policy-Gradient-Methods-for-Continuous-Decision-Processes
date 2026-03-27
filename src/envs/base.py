"""
envs/base.py
------------
Abstract base class for Markov Decision Process environments.
Defines the interface all environments must implement.

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class MDPEnv(ABC):
    """
    Abstract base class for MDP environments.

    An MDP is defined by the tuple (S, A, P, R, gamma) where:
      S     : state space
      A     : action space
      P     : transition kernel P(s' | s, a)
      R     : reward function R(s, a, s')
      gamma : discount factor in [0, 1)
    """

    def __init__(self, gamma: float = 0.99, seed: Optional[int] = None):
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self._state = None

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the state space."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action space (continuous)."""

    @property
    @abstractmethod
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (low, high) action bounds."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action; return (next_state, reward, done, info).
        """

    def render(self):
        pass

    @property
    def state(self) -> np.ndarray:
        return self._state
