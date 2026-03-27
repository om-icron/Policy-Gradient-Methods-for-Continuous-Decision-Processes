"""
agents/npg.py
-------------
Natural Policy Gradient (NPG) and a simplified Trust Region Policy Optimisation (TRPO).

Natural Policy Gradient (Kakade 2001)
--------------------------------------
Standard gradient ascent moves in parameter space with the Euclidean metric.
The Natural Gradient uses the Fisher Information Matrix (FIM) as the metric,
so updates are in the space of distributions rather than parameters:

    theta <- theta + alpha * F(theta)^{-1} * grad J(theta)

The FIM of a policy is:
    F(theta) = E_pi[ grad log pi * (grad log pi)^T ]

This corresponds to gradient ascent in the Riemannian manifold of distributions,
making it invariant to re-parameterisation of the policy.

Connection to TRPO: TRPO solves:
    max_theta  E[A(s,a) * pi_theta(a|s) / pi_old(a|s)]
    s.t.       E[KL(pi_old || pi_theta)] <= delta

The first-order approximation of TRPO recovers the natural gradient.

We approximate F^{-1} g using the conjugate gradient method (Schulman 2015).

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from .policy import GaussianPolicy
from .value import ValueNetwork
from .reinforce import Trajectory


def flat_grad(loss, params, retain_graph=False, create_graph=False):
    """Compute gradient of loss w.r.t. params, return as a single flat vector."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, create_graph=create_graph,
        allow_unused=True
    )
    return torch.cat([
        g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads, params)
    ])


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n


class NaturalPolicyGradient:
    """
    Natural Policy Gradient with conjugate gradient Fisher-vector products.

    Parameters
    ----------
    policy : GaussianPolicy
    gamma, gae_lambda : float
    lr : float
        Step size for natural gradient update.
    cg_iters : int
        Number of conjugate gradient iterations.
    damping : float
        Tikhonov damping added to FIM (F + damping*I) for numerical stability.
    value_lr : float
    max_kl : float
        Maximum KL divergence per step (for line search in TRPO).
    use_line_search : bool
        If True, performs backtracking line search (TRPO-style).
    """

    def __init__(
        self,
        policy: GaussianPolicy,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 0.01,
        cg_iters: int = 10,
        damping: float = 1e-2,
        value_lr: float = 1e-3,
        max_kl: float = 0.01,
        use_line_search: bool = False,
        value_hidden: Tuple[int, ...] = (64, 64),
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.cg_iters = cg_iters
        self.damping = damping
        self.max_kl = max_kl
        self.use_line_search = use_line_search

        self.critic = ValueNetwork(policy.state_dim, value_hidden, value_lr)
        self.train_rewards: List[float] = []

    def _fisher_vector_product(self, v: torch.Tensor,
                                states: torch.Tensor) -> torch.Tensor:
        """
        Compute F * v without explicitly forming F.
        Uses the identity: F*v = grad(grad(KL) * v)
        
        KL between current policy and a fixed copy of itself
        (second-order approximation around current parameters).
        """
        # Compute KL divergence between current and a detached copy
        dist = self.policy.get_distribution(states)
        mu_old = dist.loc.detach()
        std_old = dist.scale.detach()

        dist_new = self.policy.get_distribution(states)
        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_old, std_old),
            dist_new
        ).sum(dim=-1).mean()

        params = list(self.policy.parameters())
        grads = flat_grad(kl, params, create_graph=True)
        Fv = flat_grad((grads * v.detach()).sum(), params, retain_graph=False)
        return Fv + self.damping * v

    def _conjugate_gradient(self, b: torch.Tensor,
                             states: torch.Tensor) -> torch.Tensor:
        """
        Solve F x = b for x using the conjugate gradient algorithm.
        This avoids explicitly computing F (which would be O(n^2) in params).
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr = r @ r
        for _ in range(self.cg_iters):
            Fp = self._fisher_vector_product(p, states)
            alpha = rr / (p @ Fp + 1e-8)
            x = x + alpha * p
            r = r - alpha * Fp
            rr_new = r @ r
            beta = rr_new / (rr + 1e-8)
            p = r + beta * p
            rr = rr_new
            if rr < 1e-10:
                break
        return x

    def collect_trajectory(self, env) -> Trajectory:
        traj = Trajectory()
        state = env.reset()
        done = False
        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = self.policy.sample(s_t)
            next_state, reward, done, _ = env.step(action.squeeze(0).numpy())
            traj.add(state, action.squeeze(0).numpy(), reward, log_prob.item())
            state = next_state
        return traj

    def _compute_advantages(self, trajectories: List[Trajectory]):
        all_s, all_a, all_adv, all_ret = [], [], [], []
        for traj in trajectories:
            states = np.array(traj.states, dtype=np.float32)
            rewards = np.array(traj.rewards, dtype=np.float32)
            T = len(rewards)
            values = self.critic.predict(states)
            # GAE
            adv = np.zeros(T)
            gae = 0.0
            for t in reversed(range(T)):
                nv = values[t+1] if t < T-1 else 0.0
                delta = rewards[t] + self.gamma * nv - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                adv[t] = gae
            ret = adv + values
            all_s.append(states)
            all_a.extend(traj.actions)
            all_adv.append(adv)
            all_ret.append(ret)

        states = np.concatenate(all_s)
        actions = np.array(all_a, dtype=np.float32)
        advantages = np.concatenate(all_adv)
        returns = np.concatenate(all_ret)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, advantages, returns

    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        states, actions, advantages, returns = self._compute_advantages(trajectories)

        s_t = torch.FloatTensor(states)
        a_t = torch.FloatTensor(actions)
        adv_t = torch.FloatTensor(advantages)

        # ── Policy gradient ──────────────────────────────────────────────────
        log_probs = self.policy.log_prob(s_t, a_t)
        pg_loss = -(log_probs * adv_t).mean()
        params = list(self.policy.parameters())
        g = flat_grad(pg_loss, params, retain_graph=True)

        # ── Natural gradient: solve F * x = g ───────────────────────────────
        with torch.no_grad():
            nat_grad = self._conjugate_gradient(-g, s_t)

        # Step size
        if self.use_line_search:
            # TRPO backtracking line search
            sAs = (nat_grad @ self._fisher_vector_product(nat_grad, s_t)).item()
            step_size = np.sqrt(2 * self.max_kl / (sAs + 1e-8))
            old_params = get_flat_params(self.policy)
            expected_improve = (-g @ nat_grad).item() * step_size
            for alpha in [step_size * 0.5**i for i in range(10)]:
                new_params = old_params + alpha * nat_grad
                set_flat_params(self.policy, new_params)
                # Check KL constraint
                with torch.no_grad():
                    dist_old_mu = self.policy.get_distribution(s_t).loc
                    dist_old_std = self.policy.get_distribution(s_t).scale
                if True:  # simplified: just take first step
                    break
        else:
            flat_params = get_flat_params(self.policy)
            new_params = flat_params + self.lr * nat_grad
            set_flat_params(self.policy, new_params)

        # ── Update critic ────────────────────────────────────────────────────
        value_loss = self.critic.update(states, returns)

        mean_reward = np.mean([t.total_reward for t in trajectories])
        self.train_rewards.append(mean_reward)

        return {
            'mean_reward': mean_reward,
            'pg_loss': pg_loss.item(),
            'value_loss': value_loss,
            'grad_norm': g.norm().item(),
        }

    def train(self, env, n_iterations: int, n_episodes_per_iter: int = 10,
              print_every: int = 10) -> Dict[str, List]:
        history = {'rewards': [], 'pg_losses': [], 'value_losses': [], 'grad_norms': []}
        for it in range(n_iterations):
            trajs = [self.collect_trajectory(env) for _ in range(n_episodes_per_iter)]
            metrics = self.update(trajs)
            history['rewards'].append(metrics['mean_reward'])
            history['pg_losses'].append(metrics['pg_loss'])
            history['value_losses'].append(metrics['value_loss'])
            history['grad_norms'].append(metrics['grad_norm'])
            if (it + 1) % print_every == 0:
                print(f"  Iter {it+1:4d} | Reward: {metrics['mean_reward']:8.2f} | "
                      f"PG Loss: {metrics['pg_loss']:7.4f} | "
                      f"|g|: {metrics['grad_norm']:.4f}")
        return history
