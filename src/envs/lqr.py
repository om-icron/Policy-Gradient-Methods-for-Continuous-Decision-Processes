"""
envs/lqr.py
-----------
Linear Quadratic Regulator (LQR) environment.

System dynamics:
    x_{t+1} = A x_t + B u_t + noise

Cost (negated to reward):
    r_t = -(x_t^T Q x_t + u_t^T R u_t)

The optimal policy is a linear feedback controller:
    u* = -K* x,  K* = (R + B^T P B)^{-1} B^T P A

where P solves the Discrete Algebraic Riccati Equation (DARE).
This gives a known analytic baseline to compare against learned policies.

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from typing import Tuple, Optional
from .base import MDPEnv


class LQREnv(MDPEnv):
    """
    Discrete-time LQR environment.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector x.
    action_dim : int
        Dimension of control input u.
    A, B : np.ndarray
        System matrices. Defaults to random stable system.
    Q, R_cost : np.ndarray
        Cost matrices (must be PSD and PD respectively).
    noise_std : float
        Standard deviation of Gaussian process noise.
    max_steps : int
        Episode length.
    gamma : float
        Discount factor.
    seed : int, optional
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R_cost: Optional[np.ndarray] = None,
        noise_std: float = 0.1,
        max_steps: int = 200,
        gamma: float = 0.99,
        seed: Optional[int] = 42,
    ):
        super().__init__(gamma=gamma, seed=seed)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.noise_std = noise_std
        self.max_steps = max_steps
        self._step_count = 0

        # System matrices
        if A is None:
            # Generate a random marginally stable system (spectral radius < 1)
            rng = np.random.default_rng(seed)
            A_raw = rng.standard_normal((state_dim, state_dim))
            eigvals = np.linalg.eigvals(A_raw)
            spectral_radius = np.max(np.abs(eigvals))
            self.A = A_raw / (spectral_radius + 0.1)  # ensure stability
        else:
            self.A = np.array(A, dtype=np.float64)

        if B is None:
            rng2 = np.random.default_rng((seed or 0) + 1)
            self.B = rng2.standard_normal((state_dim, action_dim))
        else:
            self.B = np.array(B, dtype=np.float64)

        self.Q = np.eye(state_dim) if Q is None else np.array(Q, dtype=np.float64)
        self.R_cost = np.eye(action_dim) if R_cost is None else np.array(R_cost, dtype=np.float64)

        # Precompute optimal LQR gain via DARE
        self._compute_optimal_gain()

        # Action bounds: clip large controls
        self._action_low = -10.0 * np.ones(action_dim)
        self._action_high = 10.0 * np.ones(action_dim)

    def _compute_optimal_gain(self):
        """Solve DARE to get optimal LQR gain K*."""
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R_cost)
            self.K_opt = np.linalg.solve(
                self.R_cost + self.B.T @ P @ self.B,
                self.B.T @ P @ self.A
            )
            self.P_opt = P
            self._dare_solved = True
        except Exception:
            self._dare_solved = False
            self.K_opt = np.zeros((self._action_dim, self._state_dim))

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._action_low, self._action_high

    def reset(self) -> np.ndarray:
        self._state = self.rng.standard_normal(self._state_dim)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self._action_low, self._action_high)
        noise = self.rng.normal(0, self.noise_std, self._state_dim)

        next_state = self.A @ self._state + self.B @ action + noise
        reward = -(self._state @ self.Q @ self._state + action @ self.R_cost @ action)

        self._step_count += 1
        done = self._step_count >= self.max_steps

        # Clip state to prevent blow-up
        next_state = np.clip(next_state, -50.0, 50.0)
        self._state = next_state
        return next_state.copy(), float(reward), done, {}

    def optimal_action(self, state: np.ndarray) -> np.ndarray:
        """Compute u* = -K* x (optimal LQR control)."""
        return -self.K_opt @ state

    def value_function(self, state: np.ndarray) -> float:
        """
        Compute optimal value V*(x) = -x^T P* x  (undiscounted infinite-horizon).
        This is the HJB solution for LQR.
        """
        if self._dare_solved:
            return float(-state @ self.P_opt @ state)
        return 0.0
