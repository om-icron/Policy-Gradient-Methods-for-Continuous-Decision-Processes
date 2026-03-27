"""
agents/value.py
---------------
Value function approximators (critics) used as baselines in policy gradient.

Including a baseline b(s) in the policy gradient estimator:
    grad J(theta) ≈ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t))

reduces variance WITHOUT introducing bias (since E[grad log pi * b(s)] = 0).

Common choices:
  1. Constant baseline: b = mean(G)
  2. State-value function: b(s) = V^pi(s)
  3. Advantage function: A(s,a) = Q(s,a) - V(s)

We implement a neural network critic V_phi(s) trained by regression to
Monte Carlo returns.

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from .policy import MLP


class ValueNetwork(nn.Module):
    """
    Neural network value function V_phi(s).

    Trained to minimise:
        L(phi) = E[(V_phi(s) - G)^2]
    where G is the Monte Carlo return.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes=(64, 64),
        lr: float = 1e-3,
    ):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden_sizes, activation=nn.Tanh)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Numpy interface."""
        with torch.no_grad():
            s = torch.FloatTensor(states)
            return self.forward(s).numpy()

    def update(self, states: np.ndarray, targets: np.ndarray,
               n_epochs: int = 5, batch_size: int = 64) -> float:
        """
        Fit value network to target returns via mini-batch gradient descent.
        Returns average loss.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(targets),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        n_batches = 0
        for _ in range(n_epochs):
            for s_batch, g_batch in loader:
                self.optimizer.zero_grad()
                pred = self.forward(s_batch)
                loss = nn.functional.mse_loss(pred, g_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)


class LinearValueFunction:
    """
    Linear value function baseline: V(s) = w^T phi(s).

    Uses RBF features phi(s) for a nonlinear but analytically tractable
    approximation. Fitted via least-squares (ridge regression).
    """

    def __init__(self, state_dim: int, n_features: int = 100,
                 gamma_rbf: float = 0.5, ridge: float = 1e-4, seed: int = 0):
        self.state_dim = state_dim
        self.n_features = n_features
        self.gamma_rbf = gamma_rbf
        self.ridge = ridge
        rng = np.random.default_rng(seed)
        # Random RBF centres sampled from prior
        self.centres = rng.standard_normal((n_features, state_dim))
        self.weights = np.zeros(n_features)

    def _features(self, states: np.ndarray) -> np.ndarray:
        # states: (N, d) or (d,)
        if states.ndim == 1:
            states = states[np.newaxis, :]
        diff = states[:, np.newaxis, :] - self.centres[np.newaxis, :, :]  # (N, K, d)
        dist_sq = np.sum(diff**2, axis=-1)                                 # (N, K)
        return np.exp(-self.gamma_rbf * dist_sq)                           # (N, K)

    def predict(self, states: np.ndarray) -> np.ndarray:
        phi = self._features(states)
        return phi @ self.weights

    def fit(self, states: np.ndarray, targets: np.ndarray):
        phi = self._features(states)
        A = phi.T @ phi + self.ridge * np.eye(self.n_features)
        b = phi.T @ targets
        self.weights = np.linalg.solve(A, b)

    def update(self, states: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        self.fit(states, targets)
        pred = self.predict(states)
        return float(np.mean((pred - targets)**2))
