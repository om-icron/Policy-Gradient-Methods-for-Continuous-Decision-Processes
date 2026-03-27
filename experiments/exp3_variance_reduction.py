"""
experiments/exp3_variance_reduction.py
----------------------------------------
Experiment 3: Empirical Study of Gradient Variance Reduction

Goal:
    Quantify the effect of baseline subtraction on gradient estimator variance.
    This is one of the most important practical findings in policy gradient methods.

Theory:
    The REINFORCE gradient estimator is:
        g_t = grad log pi(a_t|s_t) * G_t

    Adding a baseline b(s) gives:
        g_t = grad log pi(a_t|s_t) * (G_t - b(s_t))

    The bias of the estimator is unchanged because:
        E[grad log pi(a|s) * b(s)] = b(s) * E[grad log pi(a|s)] = 0

    But the variance is reduced when b(s) ≈ V^pi(s).

We compare:
    1. No baseline
    2. Mean return baseline
    3. Neural network value baseline (fitted)

And measure:
    - Gradient SNR (signal to noise ratio)
    - Learning stability (reward variance across seeds)
    - Convergence speed

Run:
    python experiments/exp3_variance_reduction.py

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from src.envs import LQREnv, DoubleIntegrator
from src.agents import REINFORCEAgent, GaussianPolicy
from src.utils import plot_learning_curves
from src.utils.variance_analysis import (estimate_gradient_variance,
                                          plot_gradient_variance,
                                          analyse_return_distributions)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp3')
os.makedirs(RESULTS_DIR, exist_ok=True)

STATE_DIM  = 2
ACTION_DIM = 1
GAMMA      = 0.99
N_ITER     = 150
N_EPS      = 10
N_SEEDS    = 5
HIDDEN     = (32, 32)

print("=" * 60)
print("Experiment 3: Variance Reduction in Policy Gradient Methods")
print("=" * 60)

# ── Environment ───────────────────────────────────────────────────────────────
env_factory = lambda seed: DoubleIntegrator(gamma=GAMMA, seed=seed)

# ── Multi-seed training for each baseline ─────────────────────────────────────
baseline_types = ['none', 'mean', 'value_nn']
all_rewards = {bl: [] for bl in baseline_types}

for seed in range(N_SEEDS):
    print(f"\n  Seed {seed+1}/{N_SEEDS}")
    env = env_factory(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for bl in baseline_types:
        policy = GaussianPolicy(STATE_DIM, ACTION_DIM, HIDDEN)
        agent = REINFORCEAgent(policy, gamma=GAMMA, lr=3e-4, baseline=bl,
                                value_lr=1e-3, entropy_coef=0.005)
        hist = agent.train(env, N_ITER, N_EPS, print_every=N_ITER + 1)
        all_rewards[bl].append(hist['rewards'])

# ── Compute mean ± std across seeds ──────────────────────────────────────────
mean_rewards = {}
std_rewards  = {}
for bl in baseline_types:
    arr = np.array(all_rewards[bl])   # (N_SEEDS, N_ITER)
    mean_rewards[bl] = arr.mean(axis=0).tolist()
    std_rewards[bl]  = arr.std(axis=0).tolist()

# ── Plot: mean reward ± 1 std shading ────────────────────────────────────────
colors = {'none': '#e74c3c', 'mean': '#e67e22', 'value_nn': '#2ecc71'}
labels = {'none': 'No Baseline', 'mean': 'Mean Baseline', 'value_nn': 'NN Value Baseline'}

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#f8f9fa')
for bl in baseline_types:
    x = np.arange(1, N_ITER + 1)
    m = np.array(mean_rewards[bl])
    s = np.array(std_rewards[bl])
    ax.fill_between(x, m - s, m + s, alpha=0.15, color=colors[bl])
    ax.plot(x, m, color=colors[bl], linewidth=2.0, label=labels[bl])

ax.set_xlabel('Training Iteration', fontsize=12)
ax.set_ylabel('Mean Episode Reward', fontsize=12)
ax.set_title('Exp 3: Variance Reduction — Mean ± Std over 5 Seeds',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'variance_reduction_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {RESULTS_DIR}/variance_reduction_curves.png")

# ── Gradient variance estimation ──────────────────────────────────────────────
print("\nEstimating gradient variance for each baseline type...")
env = env_factory(0)
torch.manual_seed(0)

variance_results = {}
for bl in baseline_types:
    policy = GaussianPolicy(STATE_DIM, ACTION_DIM, HIDDEN)
    print(f"  Baseline: {bl}")
    res = estimate_gradient_variance(policy, env, GAMMA, n_samples=50, baseline=bl)
    variance_results[bl] = res
    print(f"    SNR = {res['snr']:.4f}, Var = {res['var_grad_norm']:.4f}, |E[g]| = {res['mean_grad_norm']:.4f}")

plot_gradient_variance(variance_results,
                        save_path=os.path.join(RESULTS_DIR, 'gradient_variance.png'))
print(f"Saved: {RESULTS_DIR}/gradient_variance.png")

# ── Gradient norm distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, bl in zip(axes, baseline_types):
    norms = variance_results[bl]['grad_norms']
    ax.hist(norms, bins=20, color=colors[bl], alpha=0.8, edgecolor='black')
    ax.set_title(f'{labels[bl]}\nSNR={variance_results[bl]["snr"]:.2f}', fontsize=11)
    ax.set_xlabel('‖∇J‖', fontsize=11)
    ax.axvline(np.mean(norms), color='black', linestyle='--', linewidth=1.5, label='Mean')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Count', fontsize=11)
fig.suptitle('Exp 3: Distribution of Gradient Norms by Baseline Type', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'gradient_norm_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Final reward variance (box plot) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
final_rewards = [all_rewards[bl][-1] for bl in baseline_types]  # last seed's last iter
all_final = [[all_rewards[bl][s][-1] for s in range(N_SEEDS)] for bl in baseline_types]
bp = ax.boxplot(all_final, labels=[labels[bl] for bl in baseline_types], patch_artist=True)
for patch, bl in zip(bp['boxes'], baseline_types):
    patch.set_facecolor(colors[bl])
    patch.set_alpha(0.7)
ax.set_ylabel('Final Episode Reward', fontsize=12)
ax.set_title('Exp 3: Final Reward Distribution Across Seeds', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'final_reward_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# Save metrics
with open(os.path.join(RESULTS_DIR, 'variance_metrics.json'), 'w') as f:
    json.dump({bl: {k: v for k, v in variance_results[bl].items() if k != 'grad_norms'}
               for bl in baseline_types}, f, indent=2)

print("\nExperiment 3 complete.")
