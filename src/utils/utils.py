"""
utils/utils.py
--------------
Utility functions for training, evaluation, and analysis.

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import Dict, List, Optional


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(agent, env, n_episodes: int = 50, deterministic: bool = True) -> Dict:
    """
    Evaluate an agent for n_episodes. Returns mean/std reward and episode lengths.
    """
    rewards, lengths = [], []
    for _ in range(n_episodes):
        state = env.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            action = agent.act(state, deterministic=deterministic)
            state, reward, done, _ = env.step(action)
            total += reward
            steps += 1
        rewards.append(total)
        lengths.append(steps)
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'all_rewards': rewards,
    }


def smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = []
    ema = values[0]
    alpha = 2.0 / (window + 1)
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return np.array(smoothed)


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    'reinforce_none': '#e74c3c',
    'reinforce_mean': '#e67e22',
    'reinforce_vnn':  '#2ecc71',
    'a2c':            '#3498db',
    'npg':            '#9b59b6',
    'linear_pg':      '#1abc9c',
    'lqr':            '#2c3e50',
    'random':         '#95a5a6',
    'pid':            '#f39c12',
}

STYLE_MAP = {
    'reinforce_none': '--',
    'reinforce_mean': '-.',
    'reinforce_vnn':  '-',
    'a2c':            '-',
    'npg':            '-',
    'linear_pg':      '-',
    'lqr':            ':',
    'random':         ':',
    'pid':            '--',
}


def plot_learning_curves(
    histories: Dict[str, List[float]],
    title: str = "Learning Curves",
    ylabel: str = "Mean Episode Reward",
    xlabel: str = "Training Iteration",
    smoothing: int = 10,
    save_path: Optional[str] = None,
    figsize=(10, 5),
    reference_lines: Optional[Dict[str, float]] = None,
):
    """
    Plot learning curves for multiple agents on the same axes.

    Parameters
    ----------
    histories : dict
        Maps agent name -> list of rewards per iteration.
    reference_lines : dict
        Maps label -> constant value (e.g., optimal LQR reward).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    for name, rewards in histories.items():
        x = np.arange(1, len(rewards) + 1)
        rewards_arr = np.array(rewards)
        smoothed = smooth(rewards, window=smoothing)
        color = COLORS.get(name, '#555555')
        ls = STYLE_MAP.get(name, '-')
        ax.plot(x, rewards_arr, alpha=0.15, color=color, linewidth=0.8)
        ax.plot(x, smoothed, color=color, linewidth=2.0, linestyle=ls,
                label=_label(name))

    if reference_lines:
        for label, val in reference_lines.items():
            ax.axhline(y=val, color='black', linestyle=':', linewidth=1.5,
                       label=label, alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_variance_comparison(
    histories: Dict[str, List[float]],
    title: str = "Return Variance Comparison",
    save_path: Optional[str] = None,
    window: int = 20,
):
    """
    Box plot showing distribution of rewards over the last `window` iterations.
    Illustrates the variance-reduction effect of baselines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
    for name, rewards in histories.items():
        tail = rewards[-window:] if len(rewards) >= window else rewards
        data.append(tail)
        labels.append(_label(name))
        colors.append(COLORS.get(name, '#555555'))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_policy_analysis(agent, env, title: str = "Policy Analysis",
                          save_path: Optional[str] = None):
    """
    For 2D state spaces: plot mean action and std as a heatmap.
    """
    import torch
    if env.state_dim != 2:
        return None

    low, high = -3.0, 3.0
    grid_size = 40
    x = np.linspace(low, high, grid_size)
    y = np.linspace(low, high, grid_size)
    X, Y = np.meshgrid(x, y)
    states = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    policy = agent.policy if hasattr(agent, 'policy') else None
    if policy is None:
        return None

    with torch.no_grad():
        s_t = torch.FloatTensor(states)
        mu, log_std = policy(s_t)
        mu_np = mu.numpy()[:, 0].reshape(grid_size, grid_size)
        std_np = log_std.exp().numpy()[:, 0].reshape(grid_size, grid_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.pcolormesh(X, Y, mu_np, cmap='RdBu_r', shading='auto')
    fig.colorbar(im1, ax=ax1, label='Mean Action')
    ax1.set_title(f'{title} — Mean Action')
    ax1.set_xlabel('State dim 1'); ax1.set_ylabel('State dim 2')

    im2 = ax2.pcolormesh(X, Y, std_np, cmap='viridis', shading='auto')
    fig.colorbar(im2, ax=ax2, label='Std Dev')
    ax2.set_title(f'{title} — Policy Std Dev')
    ax2.set_xlabel('State dim 1'); ax2.set_ylabel('State dim 2')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_hjb_comparison(lqr_env, agent, save_path: Optional[str] = None):
    """
    Compare the learned value function against the analytic HJB solution (V* = -x^T P* x)
    for the LQR environment.
    """
    import torch
    grid = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(grid, grid)
    states_2d = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    # Pad to state_dim if needed
    if lqr_env.state_dim > 2:
        pad = np.zeros((states_2d.shape[0], lqr_env.state_dim - 2))
        states_full = np.concatenate([states_2d, pad], axis=1)
    else:
        states_full = states_2d

    # Analytic V*
    analytic_V = np.array([lqr_env.value_function(s) for s in states_full])
    analytic_V = analytic_V.reshape(50, 50)

    # Learned V (from critic)
    critic = agent.critic if hasattr(agent, 'critic') else None
    if critic is not None:
        learned_V = critic.predict(states_full.astype(np.float32)).reshape(50, 50)
    else:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(analytic_V.min(), learned_V.min())
    vmax = max(analytic_V.max(), learned_V.max())

    for ax, data, title in zip(axes, [analytic_V, learned_V, analytic_V - learned_V],
                                ['Analytic V* (HJB/DARE)', 'Learned V (critic)', 'Residual V* - V_learned']):
        im = ax.pcolormesh(X, Y, data, cmap='viridis' if 'Residual' not in title else 'RdBu_r',
                           shading='auto', vmin=vmin if 'Residual' not in title else None,
                           vmax=vmax if 'Residual' not in title else None)
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('State dim 1'); ax.set_ylabel('State dim 2')

    plt.suptitle('HJB Value Function: Analytic vs Learned', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def _label(name: str) -> str:
    label_map = {
        'reinforce_none': 'REINFORCE (no baseline)',
        'reinforce_mean': 'REINFORCE (mean baseline)',
        'reinforce_vnn':  'REINFORCE (NN baseline)',
        'a2c':            'A2C (GAE)',
        'npg':            'Natural PG',
        'linear_pg':      'Linear PG (numpy)',
        'lqr':            'Optimal LQR',
        'random':         'Random Policy',
        'pid':            'PID Controller',
    }
    return label_map.get(name, name)


def print_summary_table(results: Dict[str, Dict]):
    """Pretty-print evaluation results as a comparison table."""
    print("\n" + "=" * 65)
    print(f"{'Agent':<25} {'Mean Reward':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("=" * 65)
    for name, r in results.items():
        print(f"{_label(name):<25} {r['mean_reward']:>12.2f} "
              f"{r['std_reward']:>10.2f} {r['min_reward']:>10.2f} {r['max_reward']:>10.2f}")
    print("=" * 65)
