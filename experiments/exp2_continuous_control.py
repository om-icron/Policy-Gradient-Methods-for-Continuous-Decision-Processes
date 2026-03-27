"""
experiments/exp2_continuous_control.py
---------------------------------------
Experiment 2: Continuous Control — Pendulum Swing-Up

Goal:
    Apply policy gradient methods to the continuous pendulum environment.
    This tests the framework on a nonlinear system where no analytic solution exists.
    Compare:
      - REINFORCE (NN baseline)
      - A2C (GAE)
      - Natural PG
    against PID and random baselines.

This experiment demonstrates that Gaussian policies can handle nonlinear dynamics
and that the variance reduction techniques (baselines, GAE) are critical.

Run:
    python experiments/exp2_continuous_control.py

Outputs (in results/exp2/):
    - learning_curves.png
    - policy_analysis.png
    - return_distribution.png

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json

from src.envs import ContinuousPendulum
from src.agents import (REINFORCEAgent, A2CAgent, GaussianPolicy,
                         PIDAgent, RandomAgent)
from src.agents.npg import NaturalPolicyGradient
from src.utils import evaluate_policy, plot_learning_curves, plot_variance_comparison, print_summary_table
from src.utils.utils import plot_policy_analysis

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp2')
os.makedirs(RESULTS_DIR, exist_ok=True)

STATE_DIM  = 3   # [cos(theta), sin(theta), theta_dot]
ACTION_DIM = 1
GAMMA      = 0.99
N_ITER     = 300
N_EPS      = 10
SEED       = 0
HIDDEN     = (64, 64)

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("Experiment 2: Continuous Pendulum Swing-Up")
print("=" * 60)

env = ContinuousPendulum(max_torque=2.0, dt=0.05, seed=SEED)

def make_policy():
    return GaussianPolicy(STATE_DIM, ACTION_DIM, HIDDEN, log_std_min=-3.0, log_std_max=1.0)

# ── REINFORCE ─────────────────────────────────────────────────────────────────
print("\n[1/3] Training REINFORCE (NN baseline)...")
p1 = make_policy()
agent_rein = REINFORCEAgent(p1, gamma=GAMMA, lr=3e-4, baseline='value_nn',
                             value_lr=1e-3, entropy_coef=0.01)
hist_rein = agent_rein.train(env, N_ITER, N_EPS, print_every=50)

# ── A2C ───────────────────────────────────────────────────────────────────────
print("\n[2/3] Training A2C (GAE)...")
p2 = make_policy()
agent_a2c = A2CAgent(p2, gamma=GAMMA, actor_lr=3e-4, critic_lr=1e-3,
                      entropy_coef=0.01, gae_lambda=0.95)
hist_a2c = agent_a2c.train(env, N_ITER, N_EPS, print_every=50)

# ── Natural PG ────────────────────────────────────────────────────────────────
print("\n[3/3] Training Natural PG...")
p3 = make_policy()
agent_npg = NaturalPolicyGradient(p3, gamma=GAMMA, lr=0.02, cg_iters=10,
                                    damping=1e-2, value_lr=1e-3, gae_lambda=0.95)
hist_npg = agent_npg.train(env, N_ITER, N_EPS, print_every=50)

# ── Baselines ─────────────────────────────────────────────────────────────────
pid = PIDAgent(action_dim=1, Kp=2.0, Ki=0.0, Kd=0.5, clip=2.0)
rand = RandomAgent(env.action_bounds[0], env.action_bounds[1], seed=SEED)

pid_reward = evaluate_policy(pid, env, 30, deterministic=True)['mean_reward']
rand_reward = evaluate_policy(rand, env, 30, deterministic=False)['mean_reward']
print(f"\nPID reward:    {pid_reward:.2f}")
print(f"Random reward: {rand_reward:.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
histories = {
    'reinforce_vnn': hist_rein['rewards'],
    'a2c':           hist_a2c['rewards'],
    'npg':           hist_npg['rewards'],
}

plot_learning_curves(
    histories,
    title="Exp 2: Continuous Pendulum — Learning Curves",
    reference_lines={'PID Controller': pid_reward, 'Random Policy': rand_reward},
    save_path=os.path.join(RESULTS_DIR, 'learning_curves.png'),
    smoothing=20,
)

plot_variance_comparison(
    histories,
    title="Exp 2: Return Distribution (last 40 iters)",
    save_path=os.path.join(RESULTS_DIR, 'variance_comparison.png'),
    window=40,
)

# Policy analysis: state-dependent mean and std
plot_policy_analysis(agent_a2c, env, "A2C Policy",
                     save_path=os.path.join(RESULTS_DIR, 'policy_analysis_a2c.png'))
plot_policy_analysis(agent_npg, env, "NPG Policy",
                     save_path=os.path.join(RESULTS_DIR, 'policy_analysis_npg.png'))

# ── Final evaluation ──────────────────────────────────────────────────────────
eval_results = {
    'reinforce_vnn': evaluate_policy(agent_rein.policy, env, 50, True),
    'a2c':           evaluate_policy(agent_a2c.policy, env, 50, True),
    'npg':           evaluate_policy(agent_npg.policy, env, 50, True),
    'pid':           evaluate_policy(pid, env, 50, True),
    'random':        evaluate_policy(rand, env, 50, False),
}
print_summary_table(eval_results)

with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'all_rewards'}
               for k, v in eval_results.items()}, f, indent=2)

print("\nExperiment 2 complete.")
