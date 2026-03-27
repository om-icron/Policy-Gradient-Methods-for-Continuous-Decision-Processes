"""
utils/hjb_analysis.py
---------------------
Tools for connecting policy gradient RL with stochastic optimal control
and the Hamilton-Jacobi-Bellman (HJB) equation.

The HJB equation (continuous-time, deterministic):
    0 = max_u [ f(x,u)^T grad_x V(x) + L(x,u) ] + dV/dt

For infinite-horizon discounted problems:
    rho * V(x) = max_u [ f(x,u)^T grad_x V(x) + L(x,u) ]

For LQR (f(x,u) = Ax+Bu, L(x,u) = x^TQx + u^TRu):
    The HJB solution is quadratic: V*(x) = x^T P* x
    where P* solves the Algebraic Riccati Equation (ARE):
        P*A + A^T P* - P* B R^{-1} B^T P* + Q = 0  (continuous-time)
        A^T P* A - P* - (A^T P* B)(R + B^T P* B)^{-1}(B^T P* A) + Q = 0  (discrete)

The connection to RL:
    V*(s) in RL corresponds exactly to the HJB value function.
    The optimal policy pi*(a|s) is the argmax in the HJB equation.

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are
from typing import Optional, Tuple, Dict


def compute_dare_solution(A: np.ndarray, B: np.ndarray,
                          Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Discrete Algebraic Riccati Equation.
    Returns (P*, K*) where K* = (R + B^T P* B)^{-1} B^T P* A.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def compute_care_solution(A: np.ndarray, B: np.ndarray,
                          Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Continuous Algebraic Riccati Equation.
    Returns (P*, K*) where K* = R^{-1} B^T P*.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return P, K


def bellman_residual(V_fn, env, state: np.ndarray, action: np.ndarray,
                     gamma: float) -> float:
    """
    Compute one-step Bellman residual:
        delta(s, a) = r(s,a) + gamma * V(s') - V(s)

    For the optimal value function this should be 0 at the optimal policy.
    """
    s = state.copy()
    env_copy_state = s
    next_state, reward, _, _ = env.step(action)
    residual = reward + gamma * V_fn(next_state) - V_fn(s)
    env._state = env_copy_state  # restore
    return float(residual)


def value_function_error(critic, lqr_env, n_states: int = 1000,
                         state_range: float = 3.0) -> Dict[str, float]:
    """
    Measure how well the learned critic approximates the analytic HJB solution V*.
    
    Samples random states and compares V_learned(s) vs V*(s) = -s^T P* s.
    """
    rng = np.random.default_rng(42)
    states = rng.uniform(-state_range, state_range,
                         (n_states, lqr_env.state_dim)).astype(np.float32)

    # Analytic V*
    analytic = np.array([lqr_env.value_function(s) for s in states])

    # Learned V
    learned = critic.predict(states)

    mae = float(np.mean(np.abs(analytic - learned)))
    mse = float(np.mean((analytic - learned)**2))
    rel_err = float(np.mean(np.abs(analytic - learned) / (np.abs(analytic) + 1e-8)))
    corr = float(np.corrcoef(analytic, learned)[0, 1])

    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'relative_error': rel_err,
        'correlation': corr,
    }


def policy_regret(agent, lqr_env, n_episodes: int = 50) -> Dict[str, float]:
    """
    Compute regret of learned policy vs optimal LQR:
        Regret = J(pi*) - J(pi_theta)
    
    where J(pi) is the expected return under policy pi.
    """
    from src.utils.utils import evaluate_policy

    # Evaluate learned policy
    learned = evaluate_policy(agent, lqr_env, n_episodes, deterministic=True)

    # Evaluate optimal LQR
    from src.agents.baselines import LQRAgent
    lqr_agent = LQRAgent(lqr_env)

    opt = evaluate_policy(lqr_agent, lqr_env, n_episodes, deterministic=True)

    regret = opt['mean_reward'] - learned['mean_reward']
    return {
        'optimal_reward': opt['mean_reward'],
        'learned_reward': learned['mean_reward'],
        'regret': regret,
        'regret_pct': 100.0 * regret / (abs(opt['mean_reward']) + 1e-8),
    }


def policy_gain_analysis(agent, lqr_env) -> Dict[str, np.ndarray]:
    """
    For a neural network policy, estimate the effective linear gain
    by computing K_eff = -dmu/ds at s=0.

    This lets us compare the learned policy's gain against K* from DARE.
    """
    import torch

    policy = agent.policy if hasattr(agent, 'policy') else None
    if policy is None:
        return {}

    # Linearise policy around origin using autograd
    s0 = torch.zeros(1, lqr_env.state_dim, requires_grad=True)
    mu, _ = policy(s0)
    mu_sum = mu.sum()
    grad = torch.autograd.grad(mu_sum, s0)[0]
    K_learned = -grad.detach().numpy()  # (1, state_dim) -> effective gain row

    K_opt = lqr_env.K_opt

    return {
        'K_learned': K_learned,
        'K_optimal': K_opt,
        'gain_error': float(np.linalg.norm(K_learned - K_opt[:K_learned.shape[0], :], 'fro')),
    }


def plot_value_contours(critic, lqr_env, save_path: Optional[str] = None):
    """
    Plot contours of learned V vs analytic V* in the first two state dimensions.
    """
    grid = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(grid, grid)
    states_2d = np.stack([X.ravel(), Y.ravel()], axis=1)

    if lqr_env.state_dim > 2:
        pad = np.zeros((states_2d.shape[0], lqr_env.state_dim - 2))
        states_full = np.concatenate([states_2d, pad], axis=1).astype(np.float32)
    else:
        states_full = states_2d.astype(np.float32)

    analytic_V = np.array([lqr_env.value_function(s) for s in states_full]).reshape(60, 60)
    learned_V = critic.predict(states_full).reshape(60, 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    levels = 20

    for ax, V, title in zip(axes, [analytic_V, learned_V],
                             ['Analytic V* (HJB/DARE)', 'Learned V_phi (critic)']):
        ct = ax.contourf(X, Y, V, levels=levels, cmap='viridis')
        ax.contour(X, Y, V, levels=levels, colors='white', alpha=0.2, linewidths=0.5)
        plt.colorbar(ct, ax=ax)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.plot(0, 0, 'r*', markersize=12, label='Origin (goal)')
        ax.legend()

    plt.suptitle('Value Function Comparison: Learned vs Optimal HJB Solution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
