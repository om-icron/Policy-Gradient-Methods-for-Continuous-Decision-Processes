"""
agents/policy.py
----------------
Stochastic Gaussian policies for continuous action spaces.

A Gaussian policy is:
    pi_theta(a | s) = N(mu_theta(s), sigma_theta(s)^2)

where mu_theta and sigma_theta are neural networks parameterised by theta.

The log-probability (needed for the policy gradient score function):
    log pi_theta(a | s) = -0.5 * ||( a - mu ) / sigma||^2
                          - sum(log sigma) - 0.5*d*log(2*pi)

Policy Gradient (REINFORCE) theorem:
    grad_theta J(theta) = E_tau[ sum_t grad_theta log pi_theta(a_t|s_t) * G_t ]

where G_t is the return from time t.

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional


class MLP(nn.Module):
    """Generic multi-layer perceptron."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for last layer (helps early exploration)
        last = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Stochastic Gaussian policy with state-dependent mean and log-std.

    Architecture:
        s -> MLP -> mu(s)       (mean of action distribution)
               \-> log_sigma(s) (log standard deviation)

    For simplicity we also support a *state-independent* log_std (learnable
    parameter vector), which is common in practice and more stable.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_sizes : tuple of ints
    log_std_min, log_std_max : float
        Clipping range for log std (prevents collapse / explosion).
    state_dependent_std : bool
        If True, std is predicted from state. Otherwise, a global parameter.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        log_std_min: float = -4.0,
        log_std_max: float = 2.0,
        state_dependent_std: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std

        # Mean network
        self.mu_net = MLP(state_dim, action_dim, hidden_sizes)

        if state_dependent_std:
            self.log_std_net = MLP(state_dim, action_dim, hidden_sizes)
        else:
            # Global learnable log_std (initialised to 0 → std=1)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu, log_std) for given state(s).
        """
        mu = self.mu_net(state)
        if self.state_dependent_std:
            log_std = self.log_std_net(state)
        else:
            log_std = self.log_std.expand_as(mu)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Return the action distribution N(mu, sigma^2)."""
        mu, log_std = self.forward(state)
        return Normal(mu, log_std.exp())

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and return (action, log_prob).
        Uses the reparameterisation trick: a = mu + sigma * eps, eps~N(0,I).
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        return action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log pi_theta(a|s) for given (s,a) pairs."""
        dist = self.get_distribution(state)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Entropy of the policy at given state(s): H[pi(·|s)]."""
        dist = self.get_distribution(state)
        return dist.entropy().sum(dim=-1)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Numpy interface: given a state array, return an action array.
        If deterministic=True, returns the mean action (for evaluation).
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            mu, log_std = self.forward(s)
            if deterministic:
                action = mu
            else:
                dist = Normal(mu, log_std.exp())
                action = dist.sample()
        return action.squeeze(0).numpy()


class LinearGaussianPolicy:
    """
    Linear Gaussian policy: mu(s) = W*s + b, log_std = fixed vector.

    Used as an interpretable baseline and to connect with LQR theory.
    The optimal LQR gain K* gives a deterministic linear policy u = -K*s.
    We model this as the mean of a Gaussian, shrinking std toward 0 as training
    progresses.

    This class uses numpy (no PyTorch) for analytical comparisons.
    """

    def __init__(self, state_dim: int, action_dim: int, log_std_init: float = 0.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.W = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.log_std = log_std_init * np.ones(action_dim)

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b, self.log_std])

    @params.setter
    def params(self, theta: np.ndarray):
        n = self.state_dim * self.action_dim
        self.W = theta[:n].reshape(self.action_dim, self.state_dim)
        self.b = theta[n:n + self.action_dim]
        self.log_std = theta[n + self.action_dim:]

    @property
    def num_params(self) -> int:
        return self.state_dim * self.action_dim + 2 * self.action_dim

    def mu(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def std(self) -> np.ndarray:
        return np.exp(self.log_std)

    def log_prob(self, state: np.ndarray, action: np.ndarray) -> float:
        mu = self.mu(state)
        sigma = self.std()
        return float(-0.5 * np.sum(((action - mu) / sigma)**2)
                     - np.sum(np.log(sigma))
                     - 0.5 * self.action_dim * np.log(2 * np.pi))

    def score(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Score function: grad_theta log pi_theta(a|s).
        Used in the REINFORCE policy gradient estimator.
        """
        mu = self.mu(state)
        sigma = self.std()
        residual = (action - mu) / (sigma**2)

        # Gradient w.r.t. W: outer product of residual and state
        grad_W = np.outer(residual, state)
        grad_b = residual
        # Gradient w.r.t. log_sigma: (residual^2 * sigma^2 - 1)
        grad_log_std = ((action - mu)**2 / sigma**2) - 1.0

        return np.concatenate([grad_W.ravel(), grad_b, grad_log_std])

    def sample(self, state: np.ndarray) -> np.ndarray:
        return self.mu(state) + self.std() * np.random.randn(self.action_dim)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return self.mu(state)
        return self.sample(state)
