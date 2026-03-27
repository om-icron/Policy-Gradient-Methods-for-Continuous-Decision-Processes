"""
experiments/exp1_lqr_baseline.py
---------------------------------
Experiment 1: Policy Gradient on LQR — Comparison with Analytic Optimum

Goal:
    Train REINFORCE (with various baselines), A2C, Natural PG, and a linear PG
    agent on the LQR environment. Compare performance against:
      - Optimal LQR controller (analytic gain K*)
      - Random policy
      - Zero control

This validates our implementation and gives a ground-truth comparison.
The LQR problem has a known solution, so we can compute exact regret.

Run:
    python experiments/exp1_lqr_baseline.py

Outputs (in results/exp1/):
    - learning_curves.png
    - final_comparison_table.txt
    - variance_comparison.png
    - hjb_value_contours.png

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json

from src.envs import LQREnv
from src.agents import (REINFORCEAgent, A2CAgent, GaussianPolicy,
                         RandomAgent, ZeroAgent, LQRAgent, LinearPGAgent)
from src.agents.npg import NaturalPolicyGradient
from src.utils import (evaluate_policy, plot_learning_curves, plot_variance_comparison,
                        print_summary_table)
from src.utils.hjb_analysis import value_function_error, policy_regret, plot_value_contours

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
STATE_DIM    = 4
ACTION_DIM   = 2
GAMMA        = 0.99
N_ITER       = 200
N_EPS        = 10
SEED         = 42
HIDDEN       = (64, 64)
LR_POLICY    = 3e-4
LR_VALUE     = 1e-3
ENTROPY      = 0.005

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("Experiment 1: LQR — Policy Gradient vs Analytic Optimum")
print("=" * 60)

# ── Environment ───────────────────────────────────────────────────────────────
env = LQREnv(state_dim=STATE_DIM, action_dim=ACTION_DIM, gamma=GAMMA, seed=SEED)
print(f"\nLQR Environment: state_dim={STATE_DIM}, action_dim={ACTION_DIM}")
print(f"Spectral radius of A: {max(abs(np.linalg.eigvals(env.A))):.4f}")
print(f"DARE solved: {env._dare_solved}")

# ── Agent factory ─────────────────────────────────────────────────────────────
def make_policy():
    return GaussianPolicy(STATE_DIM, ACTION_DIM, HIDDEN)

# ── REINFORCE — no baseline ───────────────────────────────────────────────────
print("\n[1/6] Training REINFORCE (no baseline)...")
p1 = make_policy()
agent_rein_none = REINFORCEAgent(p1, gamma=GAMMA, lr=LR_POLICY, baseline='none',
                                  entropy_coef=ENTROPY)
hist_rein_none = agent_rein_none.train(env, N_ITER, N_EPS, print_every=50)

# ── REINFORCE — mean baseline ─────────────────────────────────────────────────
print("\n[2/6] Training REINFORCE (mean baseline)...")
p2 = make_policy()
agent_rein_mean = REINFORCEAgent(p2, gamma=GAMMA, lr=LR_POLICY, baseline='mean',
                                  entropy_coef=ENTROPY)
hist_rein_mean = agent_rein_mean.train(env, N_ITER, N_EPS, print_every=50)

# ── REINFORCE — value NN baseline ─────────────────────────────────────────────
print("\n[3/6] Training REINFORCE (NN value baseline)...")
p3 = make_policy()
agent_rein_vnn = REINFORCEAgent(p3, gamma=GAMMA, lr=LR_POLICY, baseline='value_nn',
                                 value_lr=LR_VALUE, entropy_coef=ENTROPY)
hist_rein_vnn = agent_rein_vnn.train(env, N_ITER, N_EPS, print_every=50)

# ── A2C ───────────────────────────────────────────────────────────────────────
print("\n[4/6] Training A2C...")
p4 = make_policy()
agent_a2c = A2CAgent(p4, gamma=GAMMA, actor_lr=LR_POLICY, critic_lr=LR_VALUE,
                      entropy_coef=ENTROPY)
hist_a2c = agent_a2c.train(env, N_ITER, N_EPS, print_every=50)

# ── Natural PG ────────────────────────────────────────────────────────────────
print("\n[5/6] Training Natural Policy Gradient...")
p5 = make_policy()
agent_npg = NaturalPolicyGradient(p5, gamma=GAMMA, lr=0.05, cg_iters=10,
                                    damping=1e-2, value_lr=LR_VALUE)
hist_npg = agent_npg.train(env, N_ITER, N_EPS, print_every=50)

# ── Linear PG (numpy) ─────────────────────────────────────────────────────────
print("\n[6/6] Training Linear PG (numpy REINFORCE)...")
agent_linear = LinearPGAgent(STATE_DIM, ACTION_DIM, gamma=GAMMA, lr=5e-4, seed=SEED)
hist_linear = agent_linear.train(env, N_ITER, n_episodes_per_iter=20, print_every=50)

# ── Learning curves ───────────────────────────────────────────────────────────
histories = {
    'reinforce_none': hist_rein_none['rewards'],
    'reinforce_mean': hist_rein_mean['rewards'],
    'reinforce_vnn':  hist_rein_vnn['rewards'],
    'a2c':            hist_a2c['rewards'],
    'npg':            hist_npg['rewards'],
    'linear_pg':      hist_linear['rewards'],
}

# Evaluate optimal LQR
lqr_agent = LQRAgent(env)
opt_result = evaluate_policy(lqr_agent, env, n_episodes=50)
rand_result = evaluate_policy(RandomAgent(env.action_bounds[0], env.action_bounds[1]), env, 50, False)

print(f"\nOptimal LQR: {opt_result['mean_reward']:.2f} ± {opt_result['std_reward']:.2f}")
print(f"Random:      {rand_result['mean_reward']:.2f} ± {rand_result['std_reward']:.2f}")

plot_learning_curves(
    histories,
    title="Exp 1: LQR — Policy Gradient Learning Curves",
    reference_lines={
        'Optimal LQR (K*)': opt_result['mean_reward'],
        'Random Policy':    rand_result['mean_reward'],
    },
    save_path=os.path.join(RESULTS_DIR, 'learning_curves.png'),
    smoothing=15,
)
print(f"\nSaved: {RESULTS_DIR}/learning_curves.png")

# ── Final evaluation ──────────────────────────────────────────────────────────
eval_results = {}
for name, agent in [('reinforce_none', agent_rein_none), ('reinforce_mean', agent_rein_mean),
                     ('reinforce_vnn',  agent_rein_vnn), ('a2c', agent_a2c),
                     ('npg', agent_npg)]:
    r = evaluate_policy(agent.policy, env, 50, deterministic=True)
    eval_results[name] = r

eval_results['lqr']    = opt_result
eval_results['random'] = rand_result
print_summary_table(eval_results)

# Save summary
with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'all_rewards'}
               for k, v in eval_results.items()}, f, indent=2)

# ── Variance comparison ───────────────────────────────────────────────────────
plot_variance_comparison(
    {k: v for k, v in histories.items()},
    title="Exp 1: Return Distribution (last 20 iterations)",
    save_path=os.path.join(RESULTS_DIR, 'variance_comparison.png'),
    window=20,
)

# ── HJB / Value function comparison ──────────────────────────────────────────
if agent_rein_vnn.critic is not None and env._dare_solved:
    vf_err = value_function_error(agent_rein_vnn.critic, env)
    print(f"\nValue Function Error (REINFORCE+VNN vs analytic HJB):")
    print(f"  MAE:          {vf_err['mae']:.4f}")
    print(f"  RMSE:         {vf_err['rmse']:.4f}")
    print(f"  Rel Error:    {vf_err['relative_error']:.4f}")
    print(f"  Correlation:  {vf_err['correlation']:.4f}")

    plot_value_contours(
        agent_rein_vnn.critic, env,
        save_path=os.path.join(RESULTS_DIR, 'hjb_value_contours.png')
    )
    print(f"Saved: {RESULTS_DIR}/hjb_value_contours.png")

print("\nExperiment 1 complete.")
