"""
agents/baselines.py
-------------------
Heuristic and analytic baseline agents for comparison against learned policies.

1. RandomAgent:         Samples uniformly from action space.
2. ZeroAgent:           Always applies zero control.
3. LQRAgent:            Uses the analytic LQR optimal gain K*.
4. PIDAgent:            Proportional-integral-derivative controller.
5. LinearPGAgent:       REINFORCE with linear Gaussian policy (numpy-only).

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
from typing import Optional, List, Dict
from .policy import LinearGaussianPolicy


class RandomAgent:
    """Uniformly random policy — acts as a performance lower bound."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray, seed: int = 0):
        self.low = action_low
        self.high = action_high
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.rng.uniform(self.low, self.high)

    def train(self, env, n_iterations: int, n_episodes_per_iter: int = 10,
              print_every: int = 50) -> Dict[str, List]:
        rewards = []
        for _ in range(n_iterations):
            ep_rewards = []
            for _ in range(n_episodes_per_iter):
                s = env.reset()
                done, total = False, 0.0
                while not done:
                    a = self.act(s)
                    s, r, done, _ = env.step(a)
                    total += r
                ep_rewards.append(total)
            rewards.append(np.mean(ep_rewards))
        return {'rewards': rewards}


class ZeroAgent:
    """Zero-control agent — applies no action at every step."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(self.action_dim)

    def evaluate(self, env, n_episodes: int = 20) -> float:
        rewards = []
        for _ in range(n_episodes):
            s, done, total = env.reset(), False, 0.0
            while not done:
                s, r, done, _ = env.step(self.act(s))
                total += r
            rewards.append(total)
        return float(np.mean(rewards))


class LQRAgent:
    """
    Optimal LQR agent using the pre-computed gain K*.
    Only applicable to LQREnv — acts as the theoretical upper bound.
    """

    def __init__(self, env):
        """env must be an LQREnv with K_opt computed."""
        self.K = env.K_opt

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return -self.K @ state


class PIDAgent:
    """
    PID controller for scalar or multi-dimensional control problems.

    u_t = Kp*e_t + Ki*sum(e) + Kd*(e_t - e_{t-1})
    where e_t = -state (tracking to origin)
    """

    def __init__(self, action_dim: int, Kp: float = 1.0, Ki: float = 0.0,
                 Kd: float = 0.1, clip: float = 10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.clip = clip
        self.action_dim = action_dim
        self.integral = np.zeros(action_dim)
        self.prev_error = np.zeros(action_dim)

    def reset(self):
        self.integral = np.zeros(self.action_dim)
        self.prev_error = np.zeros(self.action_dim)

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # Error = -state (we want state -> 0)
        error = -state[:self.action_dim]
        self.integral += error
        derivative = error - self.prev_error
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return np.clip(u, -self.clip, self.clip)


class LinearPGAgent:
    """
    REINFORCE with a linear Gaussian policy (numpy-based).
    Used to demonstrate convergence on simple LQR environments and
    to validate the implementation against the analytic optimum.

    Gradient estimator (batch REINFORCE):
        g = (1/N) * sum_i sum_t [ score(s_t, a_t) * G_t ]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        self.policy = LinearGaussianPolicy(state_dim, action_dim, log_std_init=0.0)
        self.gamma = gamma
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.train_rewards: List[float] = []

    def collect_episode(self, env):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        while not done:
            action = self.policy.sample(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        return states, actions, rewards

    def compute_returns(self, rewards):
        T = len(rewards)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            G[t] = running
        return G

    def update(self, episodes):
        """episodes: list of (states, actions, rewards)."""
        gradient = np.zeros(self.policy.num_params)
        total_reward = 0.0
        n_steps = 0

        for states, actions, rewards in episodes:
            G = self.compute_returns(rewards)
            G_norm = (G - G.mean()) / (G.std() + 1e-8)
            total_reward += sum(rewards)

            for t, (s, a, g) in enumerate(zip(states, actions, G_norm)):
                score = self.policy.score(s, a)
                gradient += score * g
                n_steps += 1

        gradient /= max(len(episodes), 1)
        # Gradient ASCENT
        self.policy.params = self.policy.params + self.lr * gradient
        mean_reward = total_reward / len(episodes)
        self.train_rewards.append(mean_reward)
        return mean_reward

    def train(self, env, n_iterations: int, n_episodes_per_iter: int = 20,
              print_every: int = 50) -> Dict[str, List]:
        history = {'rewards': []}
        for it in range(n_iterations):
            episodes = [self.collect_episode(env) for _ in range(n_episodes_per_iter)]
            r = self.update(episodes)
            history['rewards'].append(r)
            if (it + 1) % print_every == 0:
                std_mean = np.exp(self.policy.log_std).mean()
                print(f"  Iter {it+1:4d} | Reward: {r:8.2f} | Std: {std_mean:.4f}")
        return history
