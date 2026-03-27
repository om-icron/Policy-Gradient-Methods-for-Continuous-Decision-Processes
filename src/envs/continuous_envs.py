"""
envs/continuous_envs.py
-----------------------
Custom continuous-action environments for policy gradient experiments.

Environments:
  1. ContinuousPendulum  — swing-up / balancing task
  2. ContinuousMountainCar — sparse reward, exploration challenge
  3. DoubleIntegrator     — simple point-mass with quadratic cost

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
from typing import Tuple, Optional
from .base import MDPEnv


class ContinuousPendulum(MDPEnv):
    """
    Underdamped pendulum with continuous torque control.

    State:  [theta, theta_dot]   (angle from upright, angular velocity)
    Action: [tau]                torque in [-max_torque, max_torque]

    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*tau^2)
    This penalises deviation from upright (theta=0) and large control effort.

    Equations of motion (Euler integration):
        theta_ddot = (g/l)*sin(theta) + tau/(m*l^2)
        theta_{t+1} = theta_t + dt * theta_dot_t
        theta_dot_{t+1} = theta_dot_t + dt * theta_ddot_t
    """

    def __init__(
        self,
        max_torque: float = 2.0,
        dt: float = 0.05,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
        max_speed: float = 8.0,
        max_steps: int = 200,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        super().__init__(gamma=gamma, seed=seed)
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.max_speed = max_speed
        self.max_steps = max_steps
        self._step_count = 0

        self._action_low = np.array([-max_torque])
        self._action_high = np.array([max_torque])

    @property
    def state_dim(self) -> int:
        return 3   # [cos(theta), sin(theta), theta_dot]  (avoids angle wrapping)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._action_low, self._action_high

    def _get_obs(self, theta: float, theta_dot: float) -> np.ndarray:
        return np.array([np.cos(theta), np.sin(theta), theta_dot])

    def reset(self) -> np.ndarray:
        # Start near bottom (theta ~ pi) with small perturbation
        theta = np.pi + self.rng.uniform(-0.1, 0.1)
        theta_dot = self.rng.uniform(-0.1, 0.1)
        self._theta = theta
        self._theta_dot = theta_dot
        self._step_count = 0
        self._state = self._get_obs(theta, theta_dot)
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        tau = float(np.clip(action[0], -self.max_torque, self.max_torque))
        theta, theta_dot = self._theta, self._theta_dot

        theta_ddot = (self.g / self.l) * np.sin(theta) + tau / (self.m * self.l**2)
        new_theta_dot = np.clip(theta_dot + self.dt * theta_ddot, -self.max_speed, self.max_speed)
        new_theta = theta + self.dt * new_theta_dot

        # Reward: negative cost (angle from upright = 0)
        angle_cost = (new_theta % (2 * np.pi) - np.pi) ** 2
        reward = -(angle_cost + 0.1 * new_theta_dot**2 + 0.001 * tau**2)

        self._theta = new_theta
        self._theta_dot = new_theta_dot
        self._state = self._get_obs(new_theta, new_theta_dot)
        self._step_count += 1
        done = self._step_count >= self.max_steps

        return self._state.copy(), float(reward), done, {"theta": new_theta}


class DoubleIntegrator(MDPEnv):
    """
    1D double integrator (point mass) with continuous force control.

    State:  [position, velocity]
    Action: [force]  in [-1, 1]

    Dynamics:
        p_{t+1} = p_t + dt * v_t
        v_{t+1} = v_t + dt * u_t

    Reward: -(p^2 + 0.1*v^2 + 0.01*u^2)   (quadratic tracking to origin)

    This is exactly an LQR problem, so the optimal policy is linear
    and we can verify learned policies against the analytic solution.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_force: float = 1.0,
        noise_std: float = 0.01,
        max_steps: int = 200,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        super().__init__(gamma=gamma, seed=seed)
        self.dt = dt
        self.max_force = max_force
        self.noise_std = noise_std
        self.max_steps = max_steps
        self._step_count = 0
        self._action_low = np.array([-max_force])
        self._action_high = np.array([max_force])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._action_low, self._action_high

    def reset(self) -> np.ndarray:
        self._state = self.rng.uniform(-1.0, 1.0, 2)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        u = float(np.clip(action[0], -self.max_force, self.max_force))
        p, v = self._state
        noise = self.rng.normal(0, self.noise_std, 2)

        new_p = p + self.dt * v + noise[0]
        new_v = v + self.dt * u + noise[1]

        reward = -(p**2 + 0.1 * v**2 + 0.01 * u**2)
        self._state = np.array([new_p, new_v])
        self._step_count += 1
        done = self._step_count >= self.max_steps
        return self._state.copy(), float(reward), done, {}
