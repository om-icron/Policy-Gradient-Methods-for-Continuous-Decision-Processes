"""
utils/variance_analysis.py
--------------------------
Tools for empirically measuring and analysing gradient variance in
policy gradient methods.

Key quantities:
  - Gradient variance (per-parameter variance of the PG estimator)
  - Signal-to-noise ratio (SNR) of the gradient estimator
  - Effective sample size
  - Baseline variance reduction ratio

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def estimate_gradient_variance(
    policy,
    env,
    gamma: float = 0.99,
    n_samples: int = 100,
    baseline: str = 'none',
    critic=None,
) -> Dict[str, float]:
    """
    Estimate the variance of the REINFORCE gradient estimator.

    Computes n_samples independent gradient estimates and measures
    their variance. This quantifies the noise in the estimator and
    explains why baseline subtraction is critical.

    Returns
    -------
    dict with keys: 'mean_grad_norm', 'var_grad_norm', 'snr', 'ess'
    """
    grad_estimates = []

    for _ in range(n_samples):
        # Collect a single episode
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.sample(s_t)
            action_np = action.squeeze(0).numpy()
            next_state, reward, done, _ = env.step(action_np)
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            state = next_state

        # Compute returns
        T = len(rewards)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            G[t] = running

        # Subtract baseline
        if baseline == 'mean':
            G = G - G.mean()
        elif baseline == 'value_nn' and critic is not None:
            states_arr = np.array(states, dtype=np.float32)
            values = critic.predict(states_arr)
            G = G - values
        G = (G - G.mean()) / (G.std() + 1e-8)

        # Compute gradient
        policy.zero_grad()
        s_tensor = torch.FloatTensor(np.array(states, dtype=np.float32))
        a_tensor = torch.FloatTensor(np.array(actions, dtype=np.float32))
        g_tensor = torch.FloatTensor(G)

        log_probs = policy.log_prob(s_tensor, a_tensor)
        loss = -(log_probs * g_tensor).mean()
        loss.backward()

        grad_flat = torch.cat([
            p.grad.data.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1)
            for p in policy.parameters()
        ]).numpy()
        grad_estimates.append(grad_flat)
        policy.zero_grad()

    grads = np.array(grad_estimates)  # (n_samples, n_params)
    mean_grad = grads.mean(axis=0)
    var_grad = grads.var(axis=0)

    grad_norms = np.linalg.norm(grads, axis=1)
    snr = np.linalg.norm(mean_grad) / (np.std(grad_norms) + 1e-8)

    return {
        'mean_grad_norm': float(np.linalg.norm(mean_grad)),
        'var_grad_norm': float(var_grad.mean()),
        'max_var': float(var_grad.max()),
        'snr': float(snr),
        'grad_norms': grad_norms.tolist(),
    }


def variance_reduction_experiment(
    policy_factory,
    env,
    gamma: float = 0.99,
    n_samples: int = 50,
) -> Dict[str, Dict]:
    """
    Run variance estimation for each baseline type and compare.

    Returns results dict mapping baseline_name -> variance metrics.
    """
    results = {}
    baselines = ['none', 'mean']
    for bl in baselines:
        print(f"  Estimating gradient variance — baseline: {bl}...")
        policy = policy_factory()
        r = estimate_gradient_variance(policy, env, gamma, n_samples, baseline=bl)
        results[bl] = r
        print(f"    |E[g]| = {r['mean_grad_norm']:.4f}, Var = {r['var_grad_norm']:.4f}, SNR = {r['snr']:.4f}")
    return results


def plot_gradient_variance(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot SNR and variance comparison across baseline types."""
    names = list(results.keys())
    snrs = [results[n]['snr'] for n in names]
    vars_ = [results[n]['var_grad_norm'] for n in names]
    labels = {
        'none': 'No Baseline',
        'mean': 'Mean Return',
        'value_nn': 'Value NN',
    }
    display = [labels.get(n, n) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ['#e74c3c', '#e67e22', '#2ecc71'][:len(names)]
    ax1.bar(display, snrs, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
    ax1.set_title('Gradient SNR by Baseline', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(display, vars_, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Mean Gradient Variance', fontsize=11)
    ax2.set_title('Gradient Variance by Baseline', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def analyse_return_distributions(
    agent,
    env,
    n_episodes: int = 200,
    gamma: float = 0.99,
) -> Dict:
    """
    Collect undiscounted and discounted returns across many episodes.
    Useful for understanding the distribution shift during training.
    """
    undiscounted, discounted = [], []
    for _ in range(n_episodes):
        state = env.reset()
        done, total_un, total_disc, t = False, 0.0, 0.0, 0
        while not done:
            action = agent.act(state, deterministic=False)
            state, reward, done, _ = env.step(action)
            total_un += reward
            total_disc += (gamma ** t) * reward
            t += 1
        undiscounted.append(total_un)
        discounted.append(total_disc)

    return {
        'undiscounted_mean': float(np.mean(undiscounted)),
        'undiscounted_std': float(np.std(undiscounted)),
        'discounted_mean': float(np.mean(discounted)),
        'discounted_std': float(np.std(discounted)),
        'undiscounted': undiscounted,
        'discounted': discounted,
    }
