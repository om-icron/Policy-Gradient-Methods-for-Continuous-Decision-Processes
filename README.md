# Policy Gradient Methods for Continuous Decision Processes

**Self Project | July 2025**

---

## Overview

This project implements and studies policy gradient reinforcement learning methods
for continuous control problems, formulated as Markov Decision Processes (MDPs).

### Key Features

- **Environments**: LQR (linear-quadratic), Continuous Pendulum, Double Integrator
- **Algorithms**: REINFORCE, A2C (GAE), Natural Policy Gradient
- **Policies**: Gaussian (neural network), Linear Gaussian (analytical)
- **Baselines**: Optimal LQR (DARE), PID, Random
- **Analysis**: Variance reduction, HJB connection, policy regret

---

## Project Structure

```
pg_project/
├── src/
│   ├── envs/
│   │   ├── base.py              # Abstract MDP base class
│   │   ├── lqr.py               # LQR environment (with DARE solution)
│   │   └── continuous_envs.py   # Pendulum, DoubleIntegrator
│   ├── agents/
│   │   ├── policy.py            # GaussianPolicy, LinearGaussianPolicy
│   │   ├── value.py             # ValueNetwork, LinearValueFunction
│   │   ├── reinforce.py         # REINFORCE + A2C agents
│   │   ├── npg.py               # Natural Policy Gradient
│   │   └── baselines.py         # LQR, PID, Random agents
│   └── utils/
│       ├── utils.py             # Evaluation, plotting
│       ├── variance_analysis.py # Gradient variance tools
│       └── hjb_analysis.py      # HJB/optimal control analysis
├── experiments/
│   ├── exp1_lqr_baseline.py     # LQR vs analytic optimum
│   ├── exp2_continuous_control.py  # Pendulum swing-up
│   ├── exp3_variance_reduction.py  # Baseline variance study
│   └── exp4_hjb_connection.py   # HJB connection analysis
├── report/
│   ├── main.tex                 # Full LaTeX report
│   └── references.bib           # Bibliography
├── results/                     # Generated figures and JSON metrics
├── run_all_experiments.py       # Master runner
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Running Experiments

```bash
# Run all experiments
python run_all_experiments.py

# Run a specific experiment
python run_all_experiments.py --exp 1
python run_all_experiments.py --exp 3 4

# Or run directly
python experiments/exp1_lqr_baseline.py
```

Results are saved to `results/exp{N}/`.

---

## Compiling the Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Mathematical Background

### Policy Gradient Theorem

For a stochastic policy π_θ:

```
∇_θ J(θ) = E_τ[ Σ_t ∇_θ log π_θ(a_t|s_t) · Q^π(s_t, a_t) ]
```

### Gaussian Policy Log-Prob

```
log π_θ(a|s) = -0.5 Σ_j ((a_j - μ_j(s)) / σ_j(s))² - Σ_j log σ_j(s) - d/2 log(2π)
```

### LQR Optimal Value (HJB Solution)

```
V*(x) = -x^T P* x    where P* solves the DARE
K* = (R + B^T P* B)^{-1} B^T P* A   (optimal gain)
u* = -K* x
```

### GAE Advantage Estimation

```
A_t^GAE(λ) = Σ_{l≥0} (γλ)^l δ_{t+l}    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

---

## Key Results Summary

| Experiment | Finding |
|-----------|---------|
| Exp 1 (LQR) | NPG achieves 2.3% regret vs optimal; no-baseline REINFORCE: 31.3% |
| Exp 2 (Pendulum) | A2C beats hand-tuned PID controller |
| Exp 3 (Variance) | NN baseline reduces gradient variance by ~60% |
| Exp 4 (HJB) | Critic converges to V*(s)=-s^T P* s with ρ=0.98 correlation |

---

## Timeline (3 months)

| Month | Activities |
|-------|-----------|
| Month 1 | MDP formulation, policy gradient derivation, REINFORCE implementation, LQR environment |
| Month 2 | A2C, Natural PG, Gaussian policy, variance analysis, pendulum environment |
| Month 3 | HJB connection, experimental evaluation, report writing, result visualisation |
