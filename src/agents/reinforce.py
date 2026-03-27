"""
agents/reinforce.py
-------------------
REINFORCE (Monte Carlo Policy Gradient) and Actor-Critic implementations.

REINFORCE (Williams 1992)
-------------------------
The policy gradient theorem states:
    grad_theta J(theta) = E_pi[ grad_theta log pi_theta(a|s) * Q^pi(s,a) ]

In REINFORCE we estimate Q^pi with Monte Carlo returns:
    G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}

Gradient estimator (with baseline):
    hat{g} = (1/N) sum_i sum_t [ grad log pi(a_t^i | s_t^i) * (G_t^i - b(s_t^i)) ]

Update rule:
    theta <- theta + alpha * hat{g}

Actor-Critic (A2C)
------------------
Instead of Monte Carlo returns, use the TD advantage:
    A_t = r_t + gamma * V(s_{t+1}) - V(s_t)

This has lower variance than MC at the cost of some bias (due to bootstrapping).

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import torch
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
from .policy import GaussianPolicy
from .value import ValueNetwork


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory buffer
# ──────────────────────────────────────────────────────────────────────────────

class Trajectory:
    """Container for a single episode trajectory."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.done: bool = False

    def add(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def compute_returns(self, gamma: float) -> np.ndarray:
        """Compute discounted returns G_t from time t to end of episode."""
        T = len(self.rewards)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = self.rewards[t] + gamma * running
            G[t] = running
        return G

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def length(self) -> int:
        return len(self.rewards)


# ──────────────────────────────────────────────────────────────────────────────
# REINFORCE Agent
# ──────────────────────────────────────────────────────────────────────────────

class REINFORCEAgent:
    """
    REINFORCE with optional baseline.

    Parameters
    ----------
    policy : GaussianPolicy
    gamma : float
        Discount factor.
    lr : float
        Policy learning rate.
    baseline : str
        'none' | 'mean' | 'value_nn' | 'value_linear'
    entropy_coef : float
        Coefficient for entropy bonus (encourages exploration).
    max_grad_norm : float
        Gradient clipping norm.
    """

    def __init__(
        self,
        policy: GaussianPolicy,
        gamma: float = 0.99,
        lr: float = 3e-4,
        baseline: str = 'value_nn',
        value_hidden: Tuple[int, ...] = (64, 64),
        value_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.baseline_type = baseline

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        # Baseline / critic
        if baseline == 'value_nn':
            self.critic = ValueNetwork(policy.state_dim, value_hidden, value_lr)
        else:
            self.critic = None

        # Logging
        self.train_rewards: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_values: List[float] = []
        self._update_count = 0

    # ── Data collection ──────────────────────────────────────────────────────

    def collect_trajectory(self, env) -> Trajectory:
        """Roll out one episode under the current policy."""
        traj = Trajectory()
        state = env.reset()
        done = False

        while not done:
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = self.policy.sample(s_tensor)
            action_np = action.squeeze(0).numpy()
            log_prob_scalar = log_prob.item()

            next_state, reward, done, _ = env.step(action_np)
            traj.add(state, action_np, reward, log_prob_scalar)
            state = next_state

        traj.done = done
        return traj

    def collect_batch(self, env, n_episodes: int) -> List[Trajectory]:
        return [self.collect_trajectory(env) for _ in range(n_episodes)]

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Perform one policy gradient update from a batch of trajectories.
        Returns a dict of training metrics.
        """
        # Collect all transitions
        all_states, all_actions, all_returns = [], [], []
        for traj in trajectories:
            G = traj.compute_returns(self.gamma)
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
            all_returns.extend(G.tolist())

        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)
        returns = np.array(all_returns, dtype=np.float32)

        # Normalise returns for gradient stability
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── Compute baseline ─────────────────────────────────────────────────
        if self.baseline_type == 'none':
            advantages = returns_norm
            value_loss = 0.0
        elif self.baseline_type == 'mean':
            advantages = returns_norm  # mean already subtracted by normalisation
            value_loss = 0.0
        elif self.baseline_type == 'value_nn':
            baseline_values = self.critic.predict(states)
            # Normalise returns to same scale as values
            advantages = returns - baseline_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            value_loss = self.critic.update(states, returns)
        else:
            advantages = returns_norm
            value_loss = 0.0

        # ── Policy gradient update ───────────────────────────────────────────
        s_t = torch.FloatTensor(states)
        a_t = torch.FloatTensor(actions)
        adv_t = torch.FloatTensor(advantages)

        # Recompute log probs with current policy (for gradient)
        log_probs = self.policy.log_prob(s_t, a_t)
        entropy = self.policy.entropy(s_t).mean()

        # REINFORCE objective: maximise E[log pi * A] + entropy_coef * H
        policy_loss = -(log_probs * adv_t).mean() - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        mean_reward = np.mean([t.total_reward for t in trajectories])
        self.train_rewards.append(mean_reward)
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss)
        self.entropy_values.append(entropy.item())
        self._update_count += 1

        return {
            'mean_reward': mean_reward,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss,
            'entropy': entropy.item(),
        }

    def train(self, env, n_iterations: int, n_episodes_per_iter: int = 10,
              print_every: int = 10) -> Dict[str, List]:
        """Full training loop."""
        history = {'rewards': [], 'policy_losses': [], 'value_losses': [], 'entropy': []}
        for it in range(n_iterations):
            trajs = self.collect_batch(env, n_episodes_per_iter)
            metrics = self.update(trajs)
            for k in history:
                key_map = {'rewards': 'mean_reward', 'policy_losses': 'policy_loss',
                           'value_losses': 'value_loss', 'entropy': 'entropy'}
                history[k].append(metrics[key_map[k]])

            if (it + 1) % print_every == 0:
                print(f"  Iter {it+1:4d} | Reward: {metrics['mean_reward']:8.2f} | "
                      f"PG Loss: {metrics['policy_loss']:7.4f} | "
                      f"VF Loss: {metrics['value_loss']:7.4f} | "
                      f"Entropy: {metrics['entropy']:.3f}")
        return history


# Need to import nn for grad clipping
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Actor-Critic (A2C) Agent
# ──────────────────────────────────────────────────────────────────────────────

class A2CAgent:
    """
    Synchronous Advantage Actor-Critic (A2C).

    Uses TD(0) advantage estimates:
        A_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

    Policy gradient:
        L_pi = -E[log pi(a|s) * A.detach()]

    Value loss:
        L_V = E[(V(s) - (r + gamma * V(s')))^2]

    Combined loss:
        L = L_pi + c_v * L_V - c_e * H[pi(·|s)]

    Parameters
    ----------
    policy : GaussianPolicy
    gamma : float
    actor_lr, critic_lr : float
    value_coef : float
        Weight on value loss.
    entropy_coef : float
        Weight on entropy bonus.
    max_grad_norm : float
    """

    def __init__(
        self,
        policy: GaussianPolicy,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        value_hidden: Tuple[int, ...] = (64, 64),
        gae_lambda: float = 0.95,
    ):
        self.policy = policy
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        self.critic = ValueNetwork(policy.state_dim, value_hidden, critic_lr)

        # Joint optimiser for actor + critic (common in A2C)
        all_params = list(policy.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(all_params, lr=actor_lr)

        self.train_rewards: List[float] = []

    def collect_trajectory(self, env) -> Trajectory:
        traj = Trajectory()
        state = env.reset()
        done = False
        while not done:
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = self.policy.sample(s_tensor)
            action_np = action.squeeze(0).numpy()
            next_state, reward, done, _ = env.step(action_np)
            traj.add(state, action_np, reward, log_prob.item())
            state = next_state
        return traj

    def _compute_gae(self, rewards, values, next_values, dones):
        """
        Generalised Advantage Estimation (GAE-lambda).
        A_t^GAE = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []

        for traj in trajectories:
            all_states.extend(traj.states[:-1] if len(traj.states) > 1 else traj.states)
            all_actions.extend(traj.actions[:-1] if len(traj.actions) > 1 else traj.actions)
            all_rewards.extend(traj.rewards[:-1] if len(traj.rewards) > 1 else traj.rewards)
            all_next_states.extend(traj.states[1:] if len(traj.states) > 1 else traj.states)
            done_flags = [False] * (len(traj.rewards) - 1) + [True]
            all_dones.extend(done_flags[:-1] if len(done_flags) > 1 else done_flags)

        if not all_states:
            return {'mean_reward': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}

        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)
        rewards = np.array(all_rewards, dtype=np.float32)
        next_states = np.array(all_next_states, dtype=np.float32)
        dones = np.array(all_dones, dtype=np.float32)

        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        s_t = torch.FloatTensor(states)
        a_t = torch.FloatTensor(actions)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)

        # Actor loss
        log_probs = self.policy.log_prob(s_t, a_t)
        entropy = self.policy.entropy(s_t).mean()
        actor_loss = -(log_probs * adv_t.detach()).mean() - self.entropy_coef * entropy

        # Critic loss
        value_pred = self.critic(s_t)
        critic_loss = nn.functional.mse_loss(value_pred, ret_t)

        # Combined loss
        total_loss = actor_loss + self.value_coef * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        mean_reward = np.mean([t.total_reward for t in trajectories])
        self.train_rewards.append(mean_reward)

        return {
            'mean_reward': mean_reward,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
        }

    def train(self, env, n_iterations: int, n_episodes_per_iter: int = 10,
              print_every: int = 10) -> Dict[str, List]:
        history = {'rewards': [], 'actor_losses': [], 'critic_losses': [], 'entropy': []}
        for it in range(n_iterations):
            trajs = [self.collect_trajectory(env) for _ in range(n_episodes_per_iter)]
            metrics = self.update(trajs)
            history['rewards'].append(metrics['mean_reward'])
            history['actor_losses'].append(metrics['actor_loss'])
            history['critic_losses'].append(metrics['critic_loss'])
            history['entropy'].append(metrics['entropy'])

            if (it + 1) % print_every == 0:
                print(f"  Iter {it+1:4d} | Reward: {metrics['mean_reward']:8.2f} | "
                      f"Actor: {metrics['actor_loss']:7.4f} | "
                      f"Critic: {metrics['critic_loss']:7.4f} | "
                      f"Entropy: {metrics['entropy']:.3f}")
        return history
