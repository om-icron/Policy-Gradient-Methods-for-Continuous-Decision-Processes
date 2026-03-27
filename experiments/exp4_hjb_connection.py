"""
experiments/exp4_hjb_connection.py
------------------------------------
Experiment 4: Connecting Policy Gradient RL with Stochastic Optimal Control

Goal:
    Explore the deep connection between policy gradient RL and the
    Hamilton-Jacobi-Bellman (HJB) equation from optimal control theory.

Theory:
    The HJB equation characterises the optimal value function V*:
        rho * V*(x) = max_u [ f(x,u)^T ∇V*(x) + L(x,u) ]

    For LQR, the HJB solution is V*(x) = -x^T P* x (quadratic form)
    where P* solves the Discrete Algebraic Riccati Equation (DARE).

    In RL, the Bellman optimality equation plays the same role:
        V*(s) = max_a [ R(s,a) + gamma * E[V*(s')] ]

    Connection: The critic V_phi(s) in Actor-Critic methods approximates
    the HJB value function. After training, we expect V_phi ≈ V* (up to
    approximation error and discount factor differences).

Experiments:
    1. Value function convergence: Plot V_phi(s) vs V*(s) = -s^T P* s
    2. Policy gain analysis: Compare learned linear approximation to K*
    3. Bellman residual: Measure how well trained critic satisfies HJB
    4. Regret analysis: Compute J(pi*) - J(pi_theta) as function of training

Run:
    python experiments/exp4_hjb_connection.py

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from src.envs import LQREnv
from src.agents import REINFORCEAgent, A2CAgent, GaussianPolicy
from src.utils.hjb_analysis import (value_function_error, policy_regret,
                                     policy_gain_analysis, plot_value_contours,
                                     compute_dare_solution)
from src.utils import evaluate_policy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp4')
os.makedirs(RESULTS_DIR, exist_ok=True)

STATE_DIM  = 2    # 2D for easier visualisation
ACTION_DIM = 1
GAMMA      = 0.99
N_ITER     = 250
N_EPS      = 15
HIDDEN     = (64, 64)
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("Experiment 4: HJB/Optimal Control Connection")
print("=" * 60)

# 2D LQR for visualisation
env = LQREnv(state_dim=STATE_DIM, action_dim=ACTION_DIM, gamma=GAMMA, seed=SEED)
print(f"\nLQR: state_dim={STATE_DIM}, action_dim={ACTION_DIM}")
print(f"Optimal gain K* = {env.K_opt}")
print(f"P* (DARE):\n{env.P_opt}")

# ── Track value function quality over training ─────────────────────────────────
policy = GaussianPolicy(STATE_DIM, ACTION_DIM, HIDDEN)
agent = REINFORCEAgent(policy, gamma=GAMMA, lr=3e-4, baseline='value_nn',
                        value_lr=1e-3, entropy_coef=0.005)

checkpoints = [10, 25, 50, 100, 150, 200, 250]
vf_errors = []
regrets = []
rewards_history = []
checkpoint_set = set(checkpoints)

from src.agents.baselines import LQRAgent
lqr_agent = LQRAgent(env)
opt_result = evaluate_policy(lqr_agent, env, 30)

print("\nTraining A2C and tracking VF quality at checkpoints...")

n_done = 0
for it in range(N_ITER):
    trajs = [agent.collect_trajectory(env) for _ in range(N_EPS)]
    metrics = agent.update(trajs)
    rewards_history.append(metrics['mean_reward'])

    if (it + 1) in checkpoint_set and agent.critic is not None:
        vf_err = value_function_error(agent.critic, env, n_states=500)
        reg = opt_result['mean_reward'] - evaluate_policy(agent.policy, env, 20, True)['mean_reward']
        vf_errors.append({'iter': it+1, **vf_err})
        regrets.append({'iter': it+1, 'regret': reg})
        print(f"  Iter {it+1:3d}: VF RMSE={vf_err['rmse']:.3f}, "
              f"Corr={vf_err['correlation']:.3f}, Regret={reg:.2f}")
    n_done = it + 1

# ── Plot 1: Regret over training ──────────────────────────────────────────────
iters_reg = [r['iter'] for r in regrets]
reg_vals  = [r['regret'] for r in regrets]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(iters_reg, reg_vals, 'o-', color='#e74c3c', linewidth=2, markersize=8)
axes[0].fill_between(iters_reg, 0, reg_vals, alpha=0.15, color='#e74c3c')
axes[0].set_xlabel('Training Iteration', fontsize=12)
axes[0].set_ylabel('Regret: J(π*) − J(π_θ)', fontsize=12)
axes[0].set_title('Policy Regret vs Optimal LQR', fontsize=12, fontweight='bold')
axes[0].axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Zero Regret')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# VF correlation
iters_vf = [v['iter'] for v in vf_errors]
corrs    = [v['correlation'] for v in vf_errors]
axes[1].plot(iters_vf, corrs, 's-', color='#3498db', linewidth=2, markersize=8)
axes[1].set_xlabel('Training Iteration', fontsize=12)
axes[1].set_ylabel('Pearson Correlation (V_φ vs V*)', fontsize=12)
axes[1].set_title('Critic Convergence to HJB Value Function', fontsize=12, fontweight='bold')
axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
axes[1].set_ylim([-0.1, 1.1])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Exp 4: Convergence to Optimal Control', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'regret_and_vf_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {RESULTS_DIR}/regret_and_vf_convergence.png")

# ── Plot 2: Value function contours (final) ───────────────────────────────────
if agent.critic is not None:
    plot_value_contours(agent.critic, env,
                         save_path=os.path.join(RESULTS_DIR, 'value_contours_final.png'))

# ── Plot 3: Policy gain analysis ──────────────────────────────────────────────
gain_info = policy_gain_analysis(agent, env)
print(f"\nPolicy gain analysis:")
print(f"  K* (optimal DARE):  {env.K_opt}")
print(f"  K_eff (learned NN): {gain_info.get('K_learned', 'N/A')}")
print(f"  Frobenius error:    {gain_info.get('gain_error', 'N/A')}")

# ── Plot 4: DARE eigenvalue analysis ─────────────────────────────────────────
P, K = compute_dare_solution(env.A, env.B, env.Q, env.R_cost)
A_cl = env.A - env.B @ K   # Closed-loop system matrix
eigvals_cl = np.linalg.eigvals(A_cl)
eigvals_A  = np.linalg.eigvals(env.A)

fig, ax = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
ax.scatter(eigvals_A.real,  eigvals_A.imag,  c='#e74c3c', s=120, zorder=5,
           label=f'Open-loop A (ρ={max(abs(eigvals_A)):.3f})', marker='x', linewidths=2)
ax.scatter(eigvals_cl.real, eigvals_cl.imag, c='#2ecc71', s=120, zorder=5,
           label=f'Closed-loop A−BK* (ρ={max(abs(eigvals_cl)):.3f})', marker='o')
ax.axhline(0, color='grey', alpha=0.4)
ax.axvline(0, color='grey', alpha=0.4)
ax.set_xlabel('Real part', fontsize=12)
ax.set_ylabel('Imaginary part', fontsize=12)
ax.set_title('Eigenvalues: Open-loop vs LQR Closed-loop System', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'eigenvalue_analysis.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {RESULTS_DIR}/eigenvalue_analysis.png")

# Save metrics
metrics_out = {
    'vf_errors': vf_errors,
    'regrets': regrets,
    'gain_analysis': {k: v.tolist() if hasattr(v, 'tolist') else v
                      for k, v in gain_info.items()},
    'eigenvalues': {
        'open_loop': {'real': eigvals_A.real.tolist(), 'imag': eigvals_A.imag.tolist()},
        'closed_loop': {'real': eigvals_cl.real.tolist(), 'imag': eigvals_cl.imag.tolist()},
    }
}
with open(os.path.join(RESULTS_DIR, 'hjb_metrics.json'), 'w') as f:
    json.dump(metrics_out, f, indent=2)

print("\nExperiment 4 complete.")
