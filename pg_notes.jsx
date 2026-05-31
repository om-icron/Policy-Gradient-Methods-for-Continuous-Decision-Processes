import { useState } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// CODE CONSTANTS  (all code examples live here to avoid JSX escaping issues)
// ═══════════════════════════════════════════════════════════════════════════

const C = {
mdpBase: `# src/envs/base.py
from abc import ABC, abstractmethod
import numpy as np

class MDPEnv(ABC):
    """
    Every environment in this project inherits from MDPEnv.
    This enforces the MDP interface: any agent can work with
    any environment without knowing the internals.
    
    Design choice: we use Python ABCs (Abstract Base Classes)
    so Python raises an error at instantiation time if a 
    subclass forgets to implement a required method.
    """
    def __init__(self, gamma: float = 0.99, seed=None):
        self.gamma = gamma                          # discount factor
        self.rng = np.random.default_rng(seed)     # reproducible randomness
        self._state = None                          # current state

    @property
    @abstractmethod
    def state_dim(self) -> int: ...    # dimensionality of S

    @property
    @abstractmethod
    def action_dim(self) -> int: ...   # dimensionality of A

    @property
    @abstractmethod
    def action_bounds(self): ...       # (low, high) arrays

    @abstractmethod
    def reset(self) -> np.ndarray: ... # sample s_0 ~ mu_0, return it

    @abstractmethod
    def step(self, action):
        # Returns: (next_state, reward, done, info)
        # This is the MDP transition kernel P and reward R combined
        ...`,

lqrEnv: `# src/envs/lqr.py  (key methods only)
class LQREnv(MDPEnv):
    def __init__(self, state_dim=4, action_dim=2, ...):
        # Build random but stable A matrix:
        # stable = spectral radius < 1 (all eigenvalues inside unit circle)
        A_raw = rng.standard_normal((state_dim, state_dim))
        rho = np.max(np.abs(np.linalg.eigvals(A_raw)))
        self.A = A_raw / (rho + 0.1)   # shrink until stable

        self.B = rng.standard_normal((state_dim, action_dim))
        self.Q = np.eye(state_dim)     # state cost: penalise deviation
        self.R_cost = np.eye(action_dim)  # control cost: penalise effort

        self._compute_optimal_gain()   # solve DARE for K*, P*

    def _compute_optimal_gain(self):
        # scipy solves the Discrete Algebraic Riccati Equation:
        # P* = Q + A^T P* A - A^T P* B (R + B^T P* B)^-1 B^T P* A
        P = solve_discrete_are(self.A, self.B, self.Q, self.R_cost)
        self.K_opt = np.linalg.solve(
            self.R_cost + self.B.T @ P @ self.B,
            self.B.T @ P @ self.A
        )  # K* = (R + B^T P* B)^-1 B^T P* A

    def step(self, action):
        noise = self.rng.normal(0, self.noise_std, self._state_dim)
        next_state = self.A @ self._state + self.B @ action + noise
        reward = -(self._state @ self.Q @ self._state    # state cost
                 + action @ self.R_cost @ action)        # control cost
        ...

    def optimal_action(self, state):
        return -self.K_opt @ state   # u* = -K* x`,

gaussianPolicy: `# src/agents/policy.py  (GaussianPolicy)
class GaussianPolicy(nn.Module):
    """
    pi_theta(a|s) = N(mu_theta(s), diag(sigma_theta(s))^2)
    
    Two sub-networks:
      mu_net   : s -> R^m   (the mean action)
      log_std  : learnable parameter vector (NOT state-dependent by default)
    
    Why not state-dependent std?
    - Simpler to train, less likely to collapse
    - Works well in practice for most control tasks
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=(64,64),
                 log_std_min=-4.0, log_std_max=2.0, state_dependent_std=False):
        super().__init__()
        self.mu_net = MLP(state_dim, action_dim, hidden_sizes)

        if state_dependent_std:
            self.log_std_net = MLP(state_dim, action_dim, hidden_sizes)
        else:
            # Global log_std: one number per action dimension
            # Initialised at 0 -> std=1 -> good initial exploration
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mu = self.mu_net(state)
        if self.state_dependent_std:
            log_std = self.log_std_net(state)
        else:
            log_std = self.log_std.expand_as(mu)
        # Clamp prevents std collapsing to 0 (no exploration) or exploding
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def log_prob(self, state, action):
        # log N(a; mu, sigma^2) = -0.5*((a-mu)/sigma)^2 - log(sigma) - 0.5*log(2pi)
        dist = self.get_distribution(state)
        return dist.log_prob(action).sum(dim=-1)  # sum over action dims

    def sample(self, state):
        # Reparameterisation: a = mu + sigma * eps, eps ~ N(0, I)
        # (makes sampling differentiable w.r.t. theta)
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob`,

mlpArch: `# src/agents/policy.py  (MLP helper)
class MLP(nn.Module):
    """
    Generic Multi-Layer Perceptron used by both actor and critic.
    Architecture: Linear -> Activation -> Linear -> ... -> Linear
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=(64,64),
                 activation=nn.Tanh):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())   # Tanh chosen: bounded, smooth
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialisation: preserves gradient norms early in training
        # gain=sqrt(2) is optimal for Tanh activations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Last layer: smaller init so initial actions are near zero
        last = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)`,

valueNet: `# src/agents/value.py  (ValueNetwork)
class ValueNetwork(nn.Module):
    """
    Critic: V_phi(s) -> single scalar (expected future return)
    
    Trained to minimise MSE loss against Monte Carlo returns:
        L(phi) = E[(V_phi(s) - G)^2]
    """
    def __init__(self, state_dim, hidden_sizes=(64,64), lr=1e-3):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden_sizes)  # output dim = 1
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        return self.net(state).squeeze(-1)  # (batch,1) -> (batch,)

    def update(self, states, targets, n_epochs=5, batch_size=64):
        """Mini-batch SGD on MSE loss. Run multiple epochs per update."""
        dataset = TensorDataset(FloatTensor(states), FloatTensor(targets))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(n_epochs):
            for s_batch, g_batch in loader:
                self.optimizer.zero_grad()
                pred = self.forward(s_batch)
                loss = F.mse_loss(pred, g_batch)
                loss.backward()
                # Clip gradients: prevents single bad batch destroying the network
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()`,

reinforce: `# src/agents/reinforce.py  (key update method)
def update(self, trajectories):
    # Step 1: collect all transitions from all episodes
    all_states, all_actions, all_returns = [], [], []
    for traj in trajectories:
        G = traj.compute_returns(self.gamma)  # G_t = sum_{k>=t} gamma^k r_k
        all_states.extend(traj.states)
        all_actions.extend(traj.actions)
        all_returns.extend(G.tolist())

    states  = np.array(all_states,  dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    returns = np.array(all_returns, dtype=np.float32)

    # Step 2: normalise returns -> stable gradient magnitudes
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Step 3: compute baseline-subtracted advantage
    if self.baseline_type == 'value_nn':
        baseline = self.critic.predict(states)       # V_phi(s)
        advantages = returns - baseline              # A = G - V(s)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.critic.update(states, returns)          # update critic too
    else:
        advantages = returns_norm

    # Step 4: policy gradient update
    s_t   = torch.FloatTensor(states)
    a_t   = torch.FloatTensor(actions)
    adv_t = torch.FloatTensor(advantages)

    # Recompute log probs WITH GRADIENT (not .no_grad() this time)
    log_probs = self.policy.log_prob(s_t, a_t)
    entropy   = self.policy.entropy(s_t).mean()

    # Negative because we MAXIMISE J (PyTorch minimises by default)
    policy_loss = -(log_probs * adv_t).mean() - self.entropy_coef * entropy

    self.optimizer.zero_grad()
    policy_loss.backward()             # backprop through log_prob
    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    self.optimizer.step()`,

gae: `# src/agents/reinforce.py  (GAE computation in A2CAgent)
def _compute_gae(self, rewards, values, next_values, dones):
    """
    Generalised Advantage Estimation (Schulman et al., 2015)
    
    delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
            = one-step TD error (surprise at time t)
    
    A_t^GAE = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
    
    lambda=0:   A = delta_t             (pure TD, low variance, high bias)
    lambda=1:   A = G_t - V(s_t)       (Monte Carlo, zero bias, high variance)
    lambda=0.95: sweet spot used in this project
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0                        # running GAE accumulator
    for t in reversed(range(T)):     # work backwards from episode end
        mask  = 1.0 - float(dones[t])
        delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
        # Accumulate: gae = delta + gamma*lambda * gae (from next step)
        gae   = delta + self.gamma * self.gae_lambda * mask * gae
        advantages[t] = gae
    returns = advantages + values    # V_target = A + V(s) (for critic update)
    return advantages, returns`,

npg: `# src/agents/npg.py  (Natural Policy Gradient — key methods)
def _fisher_vector_product(self, v, states):
    """
    Compute F*v WITHOUT forming F explicitly (F is n_params x n_params — huge).
    
    Uses the identity:
        F*v = grad( grad(KL) . v )
    
    KL here is between current policy and a copy of itself (second-order approx).
    This costs only 2 backward passes instead of n_params backward passes.
    """
    dist     = self.policy.get_distribution(states)
    mu_old   = dist.loc.detach()
    std_old  = dist.scale.detach()
    dist_new = self.policy.get_distribution(states)
    
    # KL between old (detached) and new (differentiable) distributions
    kl = kl_divergence(Normal(mu_old, std_old), dist_new).sum(-1).mean()
    
    params = list(self.policy.parameters())
    # First backward: grad(KL)
    grads = flat_grad(kl, params, create_graph=True)
    # Second backward: grad(grad(KL) . v) = F*v
    Fv = flat_grad((grads * v.detach()).sum(), params)
    return Fv + self.damping * v     # Tikhonov regularisation

def _conjugate_gradient(self, b, states):
    """
    Solve F*x = b for x using Conjugate Gradient.
    Only needs matrix-vector products F*v, never forms F.
    Converges in O(n_params) iterations in theory, 10 in practice.
    """
    x, r, p = torch.zeros_like(b), b.clone(), b.clone()
    rr = r @ r
    for _ in range(self.cg_iters):
        Fp    = self._fisher_vector_product(p, states)
        alpha = rr / (p @ Fp + 1e-8)
        x     = x + alpha * p
        r     = r - alpha * Fp
        rr_new= r @ r
        p     = r + (rr_new / (rr + 1e-8)) * p
        rr    = rr_new
        if rr < 1e-10: break
    return x`,

linearPG: `# src/agents/baselines.py  (LinearPGAgent — numpy REINFORCE)
class LinearPGAgent:
    """
    REINFORCE with a LINEAR Gaussian policy: mu(s) = W*s + b
    Implemented entirely in NumPy — no PyTorch.
    
    Purpose: on LQR, the optimal policy IS linear (u* = -K*s).
    So this agent should converge to W -> -K*.
    We use this to validate the implementation against the analytic optimum.
    """
    def score(self, state, action):
        """
        Score function: grad_theta log pi_theta(a|s)
        Derived analytically for Gaussian policy.
        """
        mu    = self.policy.mu(state)          # W*s + b
        sigma = self.policy.std()              # exp(log_std)
        resid = (action - mu) / (sigma**2)    # (a - mu) / sigma^2

        grad_W       = np.outer(resid, state) # d(log pi)/dW
        grad_b       = resid                  # d(log pi)/db
        grad_log_std = (action - mu)**2 / sigma**2 - 1.0  # d(log pi)/d(log sigma)

        return np.concatenate([grad_W.ravel(), grad_b, grad_log_std])

    def update(self, episodes):
        gradient = np.zeros(self.policy.num_params)
        for states, actions, rewards in episodes:
            G      = self.compute_returns(rewards)
            G_norm = (G - G.mean()) / (G.std() + 1e-8)
            for s, a, g in zip(states, actions, G_norm):
                gradient += self.policy.score(s, a) * g  # REINFORCE estimator
        gradient /= len(episodes)
        # ASCENT (not descent) — maximise J
        self.policy.params = self.policy.params + self.lr * gradient`,

hjbDare: `# src/utils/hjb_analysis.py  (DARE + value comparison)
from scipy.linalg import solve_discrete_are

def compute_dare_solution(A, B, Q, R):
    """
    Solves:  P = Q + A^T P A - A^T P B (R + B^T P B)^-1 B^T P A
    Returns: (P*, K*) where K* = (R + B^T P* B)^-1 B^T P* A
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K

def value_function_error(critic, lqr_env, n_states=1000):
    """
    Compare V_phi(s) against analytic V*(s) = -s^T P* s
    at random states. Measures how close RL critic is to
    the optimal HJB value function.
    """
    states   = rng.uniform(-3, 3, (n_states, lqr_env.state_dim)).astype(np.float32)
    analytic = np.array([lqr_env.value_function(s) for s in states])
    learned  = critic.predict(states)

    mae  = np.mean(np.abs(analytic - learned))
    corr = np.corrcoef(analytic, learned)[0, 1]
    return {'mae': mae, 'correlation': corr, ...}`,

expArch: `# experiments/exp1_lqr_baseline.py  (structure of every experiment)

# 1. Fix all seeds for reproducibility
torch.manual_seed(SEED); np.random.seed(SEED)

# 2. Create environment
env = LQREnv(state_dim=STATE_DIM, action_dim=ACTION_DIM, gamma=GAMMA, seed=SEED)

# 3. Factory function — same architecture for all agents
def make_policy():
    return GaussianPolicy(STATE_DIM, ACTION_DIM, HIDDEN)

# 4. Create and train each agent
agent = REINFORCEAgent(make_policy(), gamma=GAMMA, lr=LR, baseline='value_nn')
history = agent.train(env, n_iterations=200, n_episodes_per_iter=10,
                       print_every=50)

# 5. Evaluate deterministically (mean action, no sampling noise)
results = evaluate_policy(agent.policy, env, n_episodes=50, deterministic=True)

# 6. Save all outputs — figures + JSON metrics
plot_learning_curves(histories, save_path=RESULTS_DIR + '/learning_curves.png')
with open(RESULTS_DIR + '/results.json', 'w') as f:
    json.dump(results, f, indent=2)`,
};

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════

function CodeBlock({ code, lang = "python" }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try { navigator.clipboard.writeText(code); } catch (e) {}
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div style={{ margin: "1.4rem 0", borderRadius: 10, overflow: "hidden", border: "1px solid #313244" }}>
      <div style={{ background: "#181825", padding: "6px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#6c7086", fontFamily: "ui-monospace,monospace", letterSpacing: "0.06em" }}>{lang}</span>
        <button onClick={copy} style={{ fontSize: 11, color: copied ? "#a6e3a1" : "#6c7086", background: "none", border: "none", cursor: "pointer", padding: "2px 8px", borderRadius: 4, transition: "color 0.2s" }}>
          {copied ? "✓ copied" : "copy"}
        </button>
      </div>
      <pre style={{ background: "#1e1e2e", color: "#cdd6f4", margin: 0, padding: "1.1rem 1.3rem", overflowX: "auto", fontSize: 12.5, lineHeight: 1.8, fontFamily: "ui-monospace,monospace", tabSize: 4 }}>
        <code>{code}</code>
      </pre>
    </div>
  );
}

function Math({ children }) {
  return (
    <div style={{ fontFamily: "ui-monospace,monospace", fontSize: 13.5, padding: "0.8rem 1.2rem", margin: "1rem 0", background: "var(--color-background-secondary)", borderRadius: 8, borderLeft: "3px solid #8b5cf6", color: "var(--color-text-primary)", lineHeight: 1.9, whiteSpace: "pre-wrap" }}>
      {children}
    </div>
  );
}

const CTYPES = {
  insight:    { color: "#3b82f6", bg: "rgba(59,130,246,0.07)",  icon: "💡", label: "Key Insight"       },
  tricky:     { color: "#ef4444", bg: "rgba(239,68,68,0.07)",   icon: "⚠️", label: "Tricky Question"   },
  thought:    { color: "#8b5cf6", bg: "rgba(139,92,246,0.07)",  icon: "🤔", label: "Think About It"    },
  definition: { color: "#10b981", bg: "rgba(16,185,129,0.07)",  icon: "📖", label: "Definition"        },
  warning:    { color: "#f97316", bg: "rgba(249,115,22,0.07)",  icon: "⚡", label: "Common Mistake"    },
  result:     { color: "#f59e0b", bg: "rgba(245,158,11,0.07)",  icon: "📊", label: "Experiment Result" },
  arch:       { color: "#06b6d4", bg: "rgba(6,182,212,0.07)",   icon: "🏗️", label: "Architecture Note" },
};

function Callout({ type = "insight", title, children }) {
  const c = CTYPES[type] || CTYPES.insight;
  return (
    <div style={{ margin: "1.3rem 0", padding: "1rem 1.2rem", borderLeft: `3px solid ${c.color}`, background: c.bg, borderRadius: "0 8px 8px 0" }}>
      <div style={{ fontSize: 11.5, fontWeight: 700, color: c.color, marginBottom: 7, display: "flex", gap: 6, alignItems: "center", letterSpacing: "0.06em", textTransform: "uppercase" }}>
        <span>{c.icon}</span><span>{title || c.label}</span>
      </div>
      <div style={{ fontSize: 13.5, lineHeight: 1.75, color: "var(--color-text-primary)" }}>{children}</div>
    </div>
  );
}

const H1 = ({ children }) => <h1 style={{ fontSize: 27, fontWeight: 800, color: "var(--color-text-primary)", margin: "0 0 6px", lineHeight: 1.25, letterSpacing: "-0.02em" }}>{children}</h1>;
const H2 = ({ children }) => <h2 style={{ fontSize: 19, fontWeight: 700, color: "var(--color-text-primary)", margin: "2.2rem 0 0.7rem", paddingBottom: "0.5rem", borderBottom: "1px solid var(--color-border-tertiary)", lineHeight: 1.4, letterSpacing: "-0.01em" }}>{children}</h2>;
const H3 = ({ children }) => <h3 style={{ fontSize: 15.5, fontWeight: 700, color: "var(--color-text-primary)", margin: "1.7rem 0 0.4rem" }}>{children}</h3>;
const P  = ({ children }) => <p  style={{ fontSize: 13.5, lineHeight: 1.8,  color: "var(--color-text-primary)", margin: "0.55rem 0" }}>{children}</p>;
const IL = ({ children }) => <code style={{ fontFamily: "ui-monospace,monospace", fontSize: 12, padding: "1px 5px", background: "var(--color-background-secondary)", borderRadius: 4, color: "var(--color-text-primary)" }}>{children}</code>;
const Tag = ({ children, color = "#6c7086" }) => <span style={{ display: "inline-block", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", padding: "2px 8px", borderRadius: 20, background: `${color}18`, color, marginRight: 6 }}>{children}</span>;
const Divider = () => <hr style={{ border: "none", borderTop: "1px solid var(--color-border-tertiary)", margin: "2rem 0" }} />;

// ═══════════════════════════════════════════════════════════════════════════
// CHAPTER COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════

function Ch01() {
  return (
    <div>
      <Tag color="#8b5cf6">Chapter 1</Tag>
      <H1>Introduction</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>What this project is, why it matters, and how to read these notes.</em></P>
      <Divider />

      <H2>What is This Project?</H2>
      <P>This project builds a complete, from-scratch implementation of <strong>Policy Gradient Reinforcement Learning</strong> for continuous control problems. Over three months it develops from MDP fundamentals all the way to verifying that a neural network trained purely on reward signals rediscovers the same mathematical object that classical control theorists derived analytically in 1960.</P>
      <P>The project sits at an intersection of three fields:</P>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, margin: "1rem 0" }}>
        {[
          { label: "Reinforcement Learning", desc: "Learning optimal behaviour from reward signals without knowing the dynamics", color: "#3b82f6" },
          { label: "Optimal Control", desc: "Analytic methods for minimising cost in dynamical systems (LQR, HJB equation)", color: "#10b981" },
          { label: "Deep Learning", desc: "Neural networks as universal function approximators for policy and value functions", color: "#f59e0b" },
        ].map(x => (
          <div key={x.label} style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "12px 14px", borderTop: `3px solid ${x.color}` }}>
            <p style={{ fontWeight: 700, fontSize: 13, color: "var(--color-text-primary)", marginBottom: 6 }}>{x.label}</p>
            <p style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>{x.desc}</p>
          </div>
        ))}
      </div>

      <H2>The Three-Month Arc</H2>
      <P>The project was structured as three progressive phases, each building on the last:</P>
      <div style={{ display: "flex", flexDirection: "column", gap: 8, margin: "1rem 0" }}>
        {[
          { month: "Month 1", title: "Foundations", items: ["MDP formalism & environments (LQR, pendulum)", "Policy gradient theorem derivation", "REINFORCE from scratch (numpy + PyTorch)", "Analytic LQR baseline for validation"] },
          { month: "Month 2", title: "Algorithms & Extensions", items: ["Variance reduction & baselines (theory + empirical)", "A2C with GAE advantage estimation", "Gaussian policies for continuous actions", "Natural Policy Gradient with conjugate gradient"] },
          { month: "Month 3", title: "Analysis & Connections", items: ["HJB equation & Riccati connection", "Value function convergence analysis", "Systematic experiment design & evaluation", "Report writing & visualisation"] },
        ].map(p => (
          <div key={p.month} style={{ display: "flex", gap: 14, background: "var(--color-background-secondary)", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ minWidth: 80, fontSize: 11, fontWeight: 700, color: "#8b5cf6", textTransform: "uppercase", letterSpacing: "0.05em", paddingTop: 2 }}>{p.month}</div>
            <div>
              <p style={{ fontWeight: 700, fontSize: 13, color: "var(--color-text-primary)", marginBottom: 6 }}>{p.title}</p>
              <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12.5, color: "var(--color-text-secondary)", lineHeight: 1.7 }}>
                {p.items.map(i => <li key={i}>{i}</li>)}
              </ul>
            </div>
          </div>
        ))}
      </div>

      <H2>Prerequisites</H2>
      <P>These notes assume familiarity with:</P>
      <ul style={{ fontSize: 13.5, lineHeight: 1.8, color: "var(--color-text-primary)", paddingLeft: 20 }}>
        <li><strong>Linear algebra:</strong> matrix multiplication, eigenvalues, quadratic forms (xᵀAx)</li>
        <li><strong>Probability:</strong> expectations, Gaussians, conditional distributions, log-likelihood</li>
        <li><strong>Calculus:</strong> chain rule, gradients, partial derivatives</li>
        <li><strong>Python/PyTorch:</strong> neural networks, autograd, optimisers</li>
        <li><strong>Control theory (helpful but not required):</strong> state-space systems, stability</li>
      </ul>

      <H2>How to Read These Notes</H2>
      <P>Each chapter follows the same structure: <strong>plain-English intuition first</strong>, then formal math, then annotated code, then questions. The callout boxes have specific meanings:</P>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, margin: "1rem 0" }}>
        {Object.entries(CTYPES).map(([k, v]) => (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 13, color: "var(--color-text-primary)" }}>
            <span style={{ fontSize: 16 }}>{v.icon}</span>
            <strong style={{ color: v.color, minWidth: 160, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>{v.label}</strong>
            <span style={{ color: "var(--color-text-secondary)", fontSize: 12.5 }}>
              {k === "insight" && "The 'aha' moment — the core idea of a section."}
              {k === "tricky" && "A question that trips most people up, with a full answer."}
              {k === "thought" && "An open-ended question with no simple answer — for deep reflection."}
              {k === "definition" && "Formal mathematical definition."}
              {k === "warning" && "A mistake almost everyone makes when first implementing this."}
              {k === "result" && "A concrete experimental finding from the four experiments."}
              {k === "arch" && "A design decision made in the codebase and the reason why."}
            </span>
          </div>
        ))}
      </div>

      <Callout type="thought" title="Before You Start">
        Here's a question to sit with as you read: <em>If classical control theory can solve LQR problems analytically with guaranteed optimality, why bother with RL at all?</em> By Chapter 8, you'll have a precise answer — but let the question marinate.
      </Callout>
    </div>
  );
}

function Ch02() {
  return (
    <div>
      <Tag color="#3b82f6">Chapter 2</Tag>
      <H1>Markov Decision Processes</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>The formal framework that turns every control problem into the same mathematical structure.</em></P>
      <Divider />

      <H2>The Formal Definition</H2>
      <Callout type="definition" title="Markov Decision Process">
        An MDP is a tuple <strong>(S, A, P, R, γ)</strong> where:<br /><br />
        <strong>S</strong> — state space (here S ⊆ ℝⁿ, continuous)<br />
        <strong>A</strong> — action space (here A ⊆ ℝᵐ, continuous)<br />
        <strong>P(s'|s,a)</strong> — transition kernel: probability density of next state<br />
        <strong>R(s,a)</strong> — expected reward function<br />
        <strong>γ ∈ [0,1)</strong> — discount factor
      </Callout>
      <P>The <strong>Markov property</strong> is the key assumption: the next state depends only on the current state and action, not on history. Formally:</P>
      <Math>P(s_{t+1} | s_0, a_0, ..., s_t, a_t)  =  P(s_{t+1} | s_t, a_t)</Math>
      <P>This is why we only need to track the current state, not the full trajectory. It's what makes the framework computationally tractable.</P>

      <Callout type="tricky" title="What if the Markov property is violated?">
        In some real systems — like a robot with motor fatigue, or a financial market with momentum — the next state depends on history, not just the current state. You have two options: (1) augment the state to include history (e.g., stack the last k observations), or (2) use recurrent policies (LSTMs). Option 1 is what most robotics RL does. The pendulum in this project is exactly Markov: given angle and angular velocity, the physics fully determines what happens next.
      </Callout>

      <H2>The Three Environments</H2>

      <H3>1. LQREnv — the known-solution environment</H3>
      <P>Linear dynamics, quadratic cost. The entire point of this environment is that we <em>know the exact optimal solution</em> via the Discrete Algebraic Riccati Equation. This makes it ideal for validating RL implementations: if your agent can't get close to K*, something is wrong.</P>
      <Math>{"x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0, sigma^2 I)\nr_t = -(x_t^T Q x_t + u_t^T R u_t)     (negative cost = reward)"}</Math>
      <CodeBlock code={C.lqrEnv} />

      <Callout type="arch" title="Why not just use gym environments?">
        We built environments from scratch for full control over system parameters (A, B, Q, R), the ability to query the analytic optimum (K*, V*), and reproducible noise via NumPy's <IL>default_rng</IL>. Gym environments don't expose their internal structure — we need it for the HJB comparison in Experiment 4.
      </Callout>

      <H3>2. ContinuousPendulum — the nonlinear challenge</H3>
      <P>State: <IL>[cos θ, sin θ, θ̇]</IL>. Action: torque τ ∈ [−2, 2]. No analytic solution exists.</P>
      <Callout type="tricky" title="Why [cos θ, sin θ] instead of just θ?">
        If you use raw angle θ, then θ=0 and θ=2π represent the same physical configuration but look like completely different numbers to a neural network. This creates discontinuities in the policy. By encoding angle as (cos θ, sin θ), we get a smooth, periodic representation. The network sees nearby points in angle space as nearby points in input space — which is true physically. This is called <em>angle wrapping</em> and it's a common practical trick in robotics RL.
      </Callout>

      <H3>3. DoubleIntegrator — the simplest case</H3>
      <P>State: [position, velocity]. Action: force. This is actually an LQR problem in disguise, so we can verify numerical results against the analytic solution just like LQREnv, but it's simpler to visualise.</P>

      <H2>The MDP Interface in Code</H2>
      <P>Every environment inherits from <IL>MDPEnv</IL>. This abstract base class uses Python's ABC mechanism to enforce the interface — if you subclass it and forget to implement <IL>step()</IL>, Python raises a <IL>TypeError</IL> the moment you try to instantiate the class.</P>
      <CodeBlock code={C.mdpBase} />

      <Callout type="arch" title="Abstract Base Class pattern">
        The ABC pattern is one of the most important OOP patterns for research code. It means any agent can be written to work against the <IL>MDPEnv</IL> interface without knowing whether it's talking to LQREnv or ContinuousPendulum. This is the <em>Liskov Substitution Principle</em>: subtypes must be substitutable for their base types. It's what makes the experiment scripts clean — you just swap the environment object and everything else stays the same.
      </Callout>

      <H2>The Discount Factor γ</H2>
      <P>The discount factor has two interpretations:</P>
      <ul style={{ fontSize: 13.5, lineHeight: 1.8, color: "var(--color-text-primary)", paddingLeft: 20 }}>
        <li><strong>Mathematical:</strong> ensures the infinite sum <IL>Σ γᵗ rₜ</IL> converges (geometric series)</li>
        <li><strong>Economic:</strong> future rewards are worth less than immediate ones — like inflation</li>
      </ul>
      <P>With γ=0.99 (used throughout this project), a reward 100 steps away is worth 0.99¹⁰⁰ ≈ 0.37 of an immediate reward. A reward 500 steps away is worth essentially nothing (0.99⁵⁰⁰ ≈ 0.007).</P>

      <Callout type="thought" title="What happens if γ = 1?">
        The infinite sum no longer converges in general, making optimisation ill-defined. You need either finite episodes (which our environments use — max 200 steps) or a careful ergodicity argument. In practice, setting γ=1 can work for short episodes but often causes instability. What would change in the algorithms if you set γ=1 in this project?
      </Callout>
    </div>
  );
}

function Ch03() {
  return (
    <div>
      <Tag color="#10b981">Chapter 3</Tag>
      <H1>The Policy Gradient Theorem</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>The mathematical engine at the heart of the entire project.</em></P>
      <Divider />

      <H2>The Objective</H2>
      <P>We want to find policy parameters θ that maximise expected discounted return:</P>
      <Math>{"J(theta) = E_{tau ~ pi_theta} [ G(tau) ],    G(tau) = sum_{t=0}^{T} gamma^t r_t"}</Math>
      <P>The challenge: J(θ) depends on θ both through the <em>action probabilities</em> and through the <em>state distribution</em> (which states you visit depends on which actions you take). You can't just backprop through the environment.</P>

      <H2>The Log-Derivative Trick</H2>
      <P>This is the core mathematical idea. It converts an intractable gradient into a computable expectation.</P>
      <P><strong>Start:</strong> write J(θ) as an integral over trajectories:</P>
      <Math>{"J(theta) = integral p_theta(tau) G(tau) d_tau"}</Math>
      <P><strong>Take gradient:</strong></P>
      <Math>{"grad_theta J = integral G(tau) grad_theta p_theta(tau) d_tau"}</Math>
      <P><strong>Problem:</strong> <IL>grad p</IL> is not an expectation. Fix it with one identity:</P>
      <Math>{"grad f(theta) = f(theta) * grad log f(theta)\n\n=> grad p(tau) = p(tau) * grad log p(tau)"}</Math>
      <P><strong>Substitute:</strong></P>
      <Math>{"grad J = integral p_theta(tau) * grad log p_theta(tau) * G(tau) d_tau\n\n       = E_tau [ grad log p_theta(tau) * G(tau) ]"}</Math>
      <P><strong>Now it's an expectation</strong> — you can estimate it by sampling trajectories!</P>

      <Callout type="insight" title="Why the log?">
        The log appears because of the identity <IL>∇f = f · ∇log f</IL>. It converts a <em>density gradient</em> (hard to sample) into a <em>log-density gradient</em> (easy — just backprop through log π). The log also factors the trajectory probability beautifully, as we see next.
      </Callout>

      <H2>The Score Function</H2>
      <P>Now expand <IL>log p_θ(τ)</IL>:</P>
      <Math>{"log p_theta(tau) = log mu_0(s_0)              <- initial state (no theta)\n                 + sum_t log pi_theta(a_t|s_t)  <- policy (depends on theta!)\n                 + sum_t log P(s_{t+1}|s_t,a_t) <- dynamics (no theta)"}</Math>
      <P>Take gradient — first and third terms vanish:</P>
      <Math>{"grad_theta log p_theta(tau) = sum_t grad_theta log pi_theta(a_t|s_t)"}</Math>

      <Callout type="insight" title="Model-free for free">
        Notice that <IL>grad log P(s'|s,a) = 0</IL> because P doesn't depend on θ. This means <strong>we never need to know the environment dynamics</strong> to compute the policy gradient. The gradient only flows through the policy's own log-probabilities. This is why policy gradient RL is model-free.
      </Callout>

      <H2>The Full Theorem</H2>
      <Callout type="definition" title="Policy Gradient Theorem (Williams 1992, Sutton et al. 1999)">
        For a differentiable stochastic policy π_θ, the gradient of expected return is:
        <Math>{"grad_theta J(theta) = E_pi [ sum_t grad_theta log pi_theta(a_t|s_t) * Q^pi(s_t, a_t) ]"}</Math>
        The Monte Carlo (REINFORCE) estimator replaces Q^π with sampled returns:
        <Math>{"hat{grad} J = (1/N) sum_{i,t} grad log pi_theta(a_t^i | s_t^i) * G_t^i"}</Math>
      </Callout>

      <H2>Causality Reduction</H2>
      <P>One more reduction using causality: rewards at time t' &lt; t cannot be affected by action aₜ. So:</P>
      <Math>{"E[ grad log pi(a_t|s_t) * r_{t'} ] = 0   for t' < t\n\n=> use G_t = sum_{k=t}^{T} gamma^{k-t} r_k   (not the full return from t=0)"}</Math>
      <P>This halves the variance in long episodes.</P>

      <Callout type="tricky" title="Why does E[∇log π(a|s)] = 0?">
        This is the key fact behind why baselines work. Proof:
        <Math>{"E_{a~pi}[grad log pi(a|s)] = integral pi(a|s) * grad log pi(a|s) da\n                            = integral grad pi(a|s) da\n                            = grad integral pi(a|s) da\n                            = grad 1 = 0"}</Math>
        Because π is a proper probability distribution, it integrates to 1. The gradient of a constant is 0. This means any function b(s) that doesn't depend on a can be subtracted from Gₜ without changing the expected gradient — but it changes the variance. This is the mathematical foundation of baselines.
      </Callout>

      <Callout type="thought" title="What if episodes have variable length?">
        In this project all episodes have fixed length (200 steps). But in general, episodes end when <IL>done=True</IL>. If you average gradients across episodes of different lengths, should you weight by episode length? What about the discount factor — does a 10-step episode's gradient deserve equal weight to a 200-step episode's? These are subtle implementation choices that affect performance.
      </Callout>
    </div>
  );
}

function Ch04() {
  return (
    <div>
      <Tag color="#f59e0b">Chapter 4</Tag>
      <H1>Value Functions & The Critic</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>The second neural network — and why you can't train well without it.</em></P>
      <Divider />

      <H2>Definitions</H2>
      <Callout type="definition" title="State Value Function V^π(s)">
        <Math>{"V^pi(s) = E_pi [ sum_{t=0}^{inf} gamma^t r_t | s_0 = s ]\n\n       = \"expected total reward starting from state s, following pi forever\""}</Math>
      </Callout>
      <Callout type="definition" title="Action Value Function Q^π(s,a)">
        <Math>{"Q^pi(s,a) = E_pi [ sum_{t=0}^{inf} gamma^t r_t | s_0 = s, a_0 = a ]\n\n           = \"expected total reward starting from s, taking action a first, then following pi\""}</Math>
      </Callout>
      <Callout type="definition" title="Advantage Function A^π(s,a)">
        <Math>{"A^pi(s,a) = Q^pi(s,a) - V^pi(s)\n\n           = \"how much better is action a compared to what pi would do on average?\""}</Math>
        A &gt; 0 means this action was above average. A &lt; 0 means below average. A = 0 means exactly average — no gradient signal.
      </Callout>

      <H2>The Bellman Equation — Why V is Recursive</H2>
      <P>You don't need to sum infinitely many rewards to define V. It satisfies a one-step recursion:</P>
      <Math>{"V^pi(s) = E_{a~pi}[ R(s,a) + gamma * E_{s'~P}[V^pi(s')] ]"}</Math>
      <P><strong>Proof sketch:</strong> Pull out the first reward from the infinite sum, factor the expectation, and you get the value at the next state times γ.</P>

      <Callout type="insight" title="V(s) is the fixed point of the Bellman operator">
        Define the Bellman operator T^π: (T^π V)(s) = E[R + γ V(s')]. Then V^π is the unique fixed point: T^π V^π = V^π. This is why we can train the critic by minimising (V_φ(s) - (r + γ V_φ(s')))² — we're pushing V_φ toward its own fixed point. This is called <em>bootstrapping</em>.
      </Callout>

      <H2>ValueNetwork Implementation</H2>
      <P>The critic is just an MLP that takes a state and outputs a single scalar. Its loss function is pure MSE regression against Monte Carlo returns G:</P>
      <CodeBlock code={C.valueNet} />

      <Callout type="warning" title="V^π changes as π changes — staleness">
        Every time you update the actor (policy), V^π changes too — because V depends on what policy you follow. So the critic you trained last iteration is now slightly wrong. This is why you update the critic every iteration, and why using an old critic's values can introduce bias. In practice this is manageable, but it's why Actor-Critic methods can oscillate — both networks are chasing moving targets.
      </Callout>

      <H2>TD Error as Surprise</H2>
      <P>The TD error <IL>δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)</IL> has a beautiful interpretation:</P>
      <ul style={{ fontSize: 13.5, lineHeight: 1.8, color: "var(--color-text-primary)", paddingLeft: 20 }}>
        <li><IL>V(sₜ)</IL> = what I expected total reward to be from this state</li>
        <li><IL>rₜ + γV(sₜ₊₁)</IL> = what I actually got (one step of reality + revised future estimate)</li>
        <li><IL>δₜ</IL> = surprise — how wrong was my prediction?</li>
      </ul>
      <P>Positive δ means things went better than expected → the actor should repeat this action. Negative δ means things went worse → the actor should avoid it. This is exactly the advantage signal used in A2C.</P>

      <Callout type="tricky" title="In A2C, if V_φ is perfect (V_φ = V^π exactly), what is the gradient?">
        If V_φ = V^π exactly, then G - V_φ(s) = A^π(s,a) exactly. The policy gradient becomes:
        <Math>{"grad J = E[ grad log pi * A^pi(s,a) ]"}</Math>
        This is the theoretically optimal gradient — minimum variance, zero bias. In practice V_φ is always approximate, introducing some bias. The better the critic, the closer you get to this ideal. This is why training the critic well is the foundation of A2C performance.
      </Callout>

      <Callout type="thought" title="What if you used the critic to directly optimise the policy?">
        Instead of using the critic as a baseline, what if you just directly differentiated through V_φ(s) with respect to θ? This is the idea behind <em>Deterministic Policy Gradient</em> (DPG) and its successor DDPG. The trade-off: you need a differentiable (and accurate) critic, and it only works for deterministic policies. How does that compare to the stochastic approach used here?
      </Callout>
    </div>
  );
}

function Ch05() {
  return (
    <div>
      <Tag color="#ef4444">Chapter 5</Tag>
      <H1>Variance Reduction & Baselines</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>The most practically important chapter. High variance kills learning — here's how to fix it.</em></P>
      <Divider />

      <H2>Why Variance Matters</H2>
      <P>The REINFORCE estimator is unbiased: <IL>E[ĝ] = ∇J(θ)</IL>. But unbiased doesn't mean useful. If the variance is huge, you're essentially taking random walks in parameter space.</P>
      <P>Signal-to-noise ratio (SNR) measures the ratio of useful gradient signal to noise:</P>
      <Math>{"SNR = |E[g]| / std(g)"}</Math>
      <P>From Experiment 3: no baseline gives SNR = 0.63 (noise exceeds signal!). NN value baseline gives SNR = 1.48. The improvement is 2.3×.</P>

      <H2>The Zero-Bias Property</H2>
      <P>The fundamental result that makes baselines work:</P>
      <Math>{"E_{a~pi} [ grad log pi(a|s) * b(s) ] = b(s) * E[grad log pi(a|s)]\n                                              = b(s) * 0 = 0"}</Math>
      <P>So subtracting <em>any</em> function b(s) from Gₜ keeps the gradient unbiased. The variance is:</P>
      <Math>{"Var[grad log pi * (G - b)] = Var[grad log pi * G] - 2*Cov(score*G, score*b) + Var[score*b]"}</Math>
      <P>This is minimised by the optimal baseline <IL>b*(s) = E[(∇log π)²·G] / E[(∇log π)²] ≈ V^π(s)</IL>.</P>

      <H2>The Three Baselines Compared</H2>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, margin: "1rem 0" }}>
        {[
          { num: "1", name: "No Baseline", formula: "advantage = G", snr: "0.63", var: "3.87", pro: "Zero implementation cost", con: "SNR below 1 — noise dominates signal", color: "#ef4444" },
          { num: "2", name: "Mean Return", formula: "advantage = G - mean(G)", snr: "1.02", var: "2.31", pro: "One line of code", con: "Same number subtracted regardless of state difficulty", color: "#f97316" },
          { num: "3", name: "NN Value Baseline", formula: "advantage = G - V_φ(s)", snr: "1.48", var: "1.56", pro: "State-specific baseline. 60% variance reduction.", con: "Requires training a second network (the critic)", color: "#10b981" },
        ].map(b => (
          <div key={b.num} style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "12px 14px", borderLeft: `3px solid ${b.color}` }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
              <div>
                <span style={{ fontWeight: 700, fontSize: 14, color: "var(--color-text-primary)" }}>{b.name}</span>
                <span style={{ marginLeft: 10, fontFamily: "ui-monospace,monospace", fontSize: 12, color: "var(--color-text-secondary)" }}>{b.formula}</span>
              </div>
              <div style={{ textAlign: "right" }}>
                <span style={{ fontFamily: "ui-monospace,monospace", fontSize: 13, fontWeight: 700, color: b.color }}>SNR {b.snr}</span>
                <span style={{ fontSize: 11, color: "var(--color-text-secondary)", marginLeft: 8 }}>Var {b.var}</span>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <p style={{ fontSize: 12, color: "#10b981", lineHeight: 1.5 }}>✓ {b.pro}</p>
              <p style={{ fontSize: 12, color: "#ef4444", lineHeight: 1.5 }}>✗ {b.con}</p>
            </div>
          </div>
        ))}
      </div>

      <H2>Generalised Advantage Estimation (GAE)</H2>
      <P>GAE (Schulman et al., 2015) solves the bias-variance trade-off for the advantage estimate itself. Instead of waiting for the full Monte Carlo return, it blends TD errors:</P>
      <CodeBlock code={C.gae} />
      <P>The λ parameter controls the trade-off:</P>
      <Math>{"lambda = 0:  A = r_t + gamma*V(s_{t+1}) - V(s_t)   (pure TD, low variance, biased)\nlambda = 1:  A = G_t - V(s_t)                       (Monte Carlo, zero bias, high variance)\nlambda = 0.95: used in this project (near-MC, low bias)"}</Math>

      <Callout type="result" title="Experiment 3 Key Numbers">
        Measured empirically over 5 seeds × 50 gradient samples on the Double Integrator environment:
        <ul style={{ paddingLeft: 16, margin: "8px 0" }}>
          <li>No baseline: gradient variance 3.87, SNR 0.63</li>
          <li>Mean baseline: gradient variance 2.31 (−40%), SNR 1.02</li>
          <li>NN value baseline: gradient variance 1.56 (−60%), SNR 1.48</li>
        </ul>
        The 2.3× SNR improvement directly translates to faster, more stable convergence.
      </Callout>

      <Callout type="thought" title="Can variance be reduced further?">
        Yes — and this is an active research area. Techniques include: (1) <em>control variates</em> using multiple rollouts, (2) <em>importance sampling</em> to reuse old data, (3) <em>variance reduction via value decomposition</em>. Modern methods like PPO and SAC each have their own variance reduction tricks. How would you design an experiment to test whether further reduction actually helps on the pendulum?
      </Callout>
    </div>
  );
}

function Ch06() {
  return (
    <div>
      <Tag color="#06b6d4">Chapter 6</Tag>
      <H1>Gaussian Policies for Continuous Actions</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>Why Gaussian, how to parameterise it, and what the code actually does.</em></P>
      <Divider />

      <H2>Why Gaussian?</H2>
      <P>For discrete action spaces, a softmax output gives valid probabilities. For continuous actions (like torque ∈ [−2, 2]), we need a <em>distribution over ℝᵐ</em>. The Gaussian is the natural choice:</P>
      <ul style={{ fontSize: 13.5, lineHeight: 1.8, color: "var(--color-text-primary)", paddingLeft: 20 }}>
        <li><strong>Analytically tractable:</strong> log-prob, entropy, KL divergence all have closed forms</li>
        <li><strong>Differentiable:</strong> log π(a|s) is smooth in both a and the parameters</li>
        <li><strong>Exploration built in:</strong> σ controls how much you explore</li>
        <li><strong>Works for LQR:</strong> the optimal policy u* = −K*x is the mean of a Gaussian with σ→0</li>
      </ul>

      <H2>The Parameterisation</H2>
      <Math>{"pi_theta(a|s) = N(a | mu_theta(s), diag(sigma_theta(s))^2)\n\nmu_theta(s)    : S -> R^m    (neural network — the mean action)\nsigma_theta(s) : S -> R^m+  (positive std dev — controls exploration)"}</Math>
      <P>The policy outputs a <em>distribution</em>, not a single action. At inference time you either sample from it (stochastic, for training) or take the mean (deterministic, for evaluation).</P>

      <H2>The Log-Probability Formula</H2>
      <P>This is what you need for the policy gradient. For a diagonal Gaussian:</P>
      <Math>{"log pi_theta(a|s) = -0.5 * sum_j ((a_j - mu_j(s)) / sigma_j(s))^2\n                   - sum_j log(sigma_j(s))\n                   - (m/2) * log(2*pi)"}</Math>
      <P>The key term is <IL>(a−μ)/σ</IL> — the standardised residual. If action a is far from mean μ, log-prob is very negative. If a is exactly at the mean, the squared term is 0 and log-prob is maximised.</P>

      <H2>Implementation: GaussianPolicy</H2>
      <CodeBlock code={C.gaussianPolicy} />

      <Callout type="arch" title="State-independent log_std: the design choice">
        The code uses a global learnable <IL>log_std</IL> parameter (not state-dependent). This is simpler and more stable. With state-dependent std, the network can learn to collapse σ to 0 in some states — which kills exploration and can cause NaN gradients. The global parameter is clipped to [log_std_min, log_std_max] = [−4, 2], keeping σ ∈ [0.018, 7.4]. This range allows useful exploration without numerical explosions.
      </Callout>

      <H2>MLP Architecture</H2>
      <CodeBlock code={C.mlpArch} />

      <Callout type="tricky" title="Why Tanh activation for a control policy?">
        Three reasons: (1) Tanh outputs are bounded (−1, 1), preventing extreme pre-activations. (2) It's smooth everywhere — better gradient flow than ReLU for RL where gradients are already noisy. (3) For LQR, the optimal policy is <em>linear</em> in state — Tanh with small weights approximates a linear function near the origin, which is where the optimal policy operates. ReLU would break this symmetry.
      </Callout>

      <H2>The Reparameterisation Trick</H2>
      <P>Sampling <IL>a ~ N(μ,σ²)</IL> can be written as:</P>
      <Math>{"a = mu(s) + sigma(s) * eps,    eps ~ N(0, I)"}</Math>
      <P>This makes sampling differentiable w.r.t. θ — gradients can flow through the sampled action into μ and σ. REINFORCE uses the score function trick instead (doesn't need this), but algorithms like SAC rely entirely on reparameterisation. The project implements both approaches in the codebase.</P>

      <Callout type="thought" title="What distribution would be better for bounded action spaces?">
        The Gaussian has support on all of ℝ, but our torque is bounded in [−2, 2]. This means we clip sampled actions, which introduces a discontinuity. Alternatives: (1) Beta distribution (support [0,1], rescale), (2) Tanh-squashed Gaussian (used in SAC), (3) truncated Gaussian. Each has different properties for the score function gradient. Which would you use and why?
      </Callout>
    </div>
  );
}

function Ch07() {
  return (
    <div>
      <Tag color="#8b5cf6">Chapter 7</Tag>
      <H1>The Three Algorithms</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>REINFORCE, A2C, and Natural PG — same goal, different mechanics.</em></P>
      <Divider />

      <H2>Algorithm 1: REINFORCE</H2>
      <P>The simplest policy gradient algorithm. Collect full episodes, compute returns G, update the policy with the gradient estimator.</P>
      <CodeBlock code={C.reinforce} />

      <Callout type="warning" title="The double gradient computation bug">
        Notice the comment "Recompute log probs WITH GRADIENT." During <IL>collect_trajectory</IL>, we use <IL>torch.no_grad()</IL> for efficiency — we don't need gradients when just acting. But during the update, we <em>must</em> recompute log probs with gradients enabled so that <IL>policy_loss.backward()</IL> works. A common bug is reusing the log probs from collection — they have no gradient graph and <IL>.backward()</IL> silently does nothing.
      </Callout>

      <H2>Algorithm 2: A2C with GAE</H2>
      <P>A2C adds three improvements over REINFORCE: (1) GAE advantages instead of MC returns, (2) a simultaneously trained critic, (3) an entropy bonus to prevent premature convergence.</P>
      <P>The combined loss is:</P>
      <Math>{"L = L_actor + c_V * L_critic - c_H * H[pi]\n\nL_actor  = -E[ log pi(a|s) * A_GAE ]   (maximise advantage-weighted log-prob)\nL_critic = E[ (V_phi(s) - G)^2 ]       (minimise critic prediction error)\nH[pi]    = E[ -log pi(a|s) ]            (entropy bonus: stay exploratory)\n\nc_V = 0.5,  c_H = 0.01 in this project"}</Math>
      <CodeBlock code={C.gae} />

      <Callout type="insight" title="Why the entropy bonus?">
        Without the entropy term, the policy can collapse to a deterministic policy prematurely — it finds a local optimum and the std σ shrinks to near-zero. Once σ ≈ 0, exploration stops and you're stuck. The entropy bonus <IL>H[π] = Σlog σ + const</IL> penalises small σ, keeping the policy exploratory throughout training. The coefficient c_H = 0.01 is small enough not to dominate but large enough to prevent collapse.
      </Callout>

      <H2>Algorithm 3: Natural Policy Gradient</H2>
      <P>The key insight: standard gradient ascent takes equal steps in <em>parameter space</em>, but equal parameter changes can cause wildly different changes in the <em>policy distribution</em>. NPG corrects this by stepping in distribution space.</P>
      <Math>{"Standard gradient:  theta <- theta + alpha * grad J\nNatural gradient:   theta <- theta + alpha * F^{-1} * grad J\n\nF = E[ grad log pi * (grad log pi)^T ]   (Fisher Information Matrix)"}</Math>
      <P>F^{-1} is never computed explicitly (it's n_params × n_params — huge). Instead, we solve <IL>Fx = g</IL> using conjugate gradient, which only needs matrix-vector products <IL>Fv</IL>:</P>
      <CodeBlock code={C.npg} />

      <Callout type="tricky" title="What's the connection between NPG and Newton's method?">
        In optimisation, Newton's method steps as <IL>θ ← θ - H⁻¹∇L</IL> where H is the Hessian of the loss. The Natural Gradient replaces H with F (Fisher Information Matrix). For exponential family distributions, F = −E[H log p] — so they're deeply related. Both use second-order information to adapt step sizes. The key difference: Newton uses the curvature of the loss landscape; NPG uses the curvature of the distribution manifold. Which is more appropriate for RL?
      </Callout>

      <Callout type="result" title="Experiment 1 Results">
        After 200 iterations on 4D LQR:
        <ul style={{ paddingLeft: 16, margin: "8px 0" }}>
          <li>Optimal LQR (K*): −312.4 reward (the ceiling)</li>
          <li>Natural PG: −319.7 (2.3% regret)</li>
          <li>A2C: −323.5 (3.6% regret)</li>
          <li>REINFORCE + NN baseline: −331.2 (6% regret)</li>
          <li>REINFORCE no baseline: −410.3 (31% regret)</li>
        </ul>
        NPG achieves the lowest regret despite the same number of training steps — because it wastes fewer steps on poorly-scaled gradient directions.
      </Callout>

      <H2>Comparing the Three</H2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, margin: "1rem 0" }}>
        {[
          { name: "REINFORCE", variance: "High", bias: "Zero", complexity: "O(n)", best: "Simple problems, debugging", color: "#3b82f6" },
          { name: "A2C + GAE", variance: "Low", bias: "Small (λ<1)", complexity: "O(n)", best: "General use, most problems", color: "#10b981" },
          { name: "Natural PG", variance: "Low", bias: "Small", complexity: "O(n) per iter + CG", best: "When convergence speed matters", color: "#8b5cf6" },
        ].map(a => (
          <div key={a.name} style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "12px 14px", borderTop: `3px solid ${a.color}` }}>
            <p style={{ fontWeight: 700, fontSize: 13, color: "var(--color-text-primary)", marginBottom: 8 }}>{a.name}</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {[["Variance", a.variance], ["Bias", a.bias], ["Per-iter cost", a.complexity], ["Best for", a.best]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: "var(--color-text-secondary)" }}>{k}</span>
                  <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Ch08() {
  return (
    <div>
      <Tag color="#f97316">Chapter 8</Tag>
      <H1>HJB Equation & Optimal Control</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>The deepest result: RL and control theory are solving the same problem.</em></P>
      <Divider />

      <H2>The Hamilton-Jacobi-Bellman Equation</H2>
      <P>Stochastic optimal control characterises the optimal value function V* via the HJB PDE:</P>
      <Math>{"rho * V*(x) = max_u [ L(x,u) + f(x,u)^T grad_x V*(x) + 0.5 tr(Sigma grad^2_x V*(x)) ]\n\nwhere: f(x,u) = system dynamics (drift)\n       L(x,u) = running cost\n       Sigma   = noise covariance\n       rho     = discount rate"}</Math>
      <P>This is a nonlinear PDE in general — hard to solve. But for LQR it collapses to a simple algebraic equation.</P>

      <H2>LQR as a Special Case</H2>
      <P>For LQR with <IL>f(x,u) = Ax+Bu</IL> and <IL>L(x,u) = xᵀQx + uᵀRu</IL>, the HJB solution is quadratic:</P>
      <Math>{"V*(x) = -x^T P* x     (quadratic bowl centred at origin)"}</Math>
      <P>where P* solves the Discrete Algebraic Riccati Equation (DARE):</P>
      <Math>{"P* = Q + A^T P* A - A^T P* B (R + B^T P* B)^{-1} B^T P* A"}</Math>
      <P>The optimal policy recovered from V* is linear:</P>
      <Math>{"u*(x) = argmin_u [L(x,u) + (Ax+Bu)^T grad V*(x)]\n       = -(R + B^T P* B)^{-1} B^T P* A x\n       = -K* x"}</Math>
      <CodeBlock code={C.hjbDare} />

      <H2>The RL–Control Correspondence</H2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, background: "var(--color-border-tertiary)", borderRadius: 8, overflow: "hidden", margin: "1rem 0" }}>
        <div style={{ background: "var(--color-background-secondary)", padding: "10px 14px", fontWeight: 700, fontSize: 12, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Optimal Control</div>
        <div style={{ background: "var(--color-background-secondary)", padding: "10px 14px", fontWeight: 700, fontSize: 12, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Reinforcement Learning</div>
        {[
          ["HJB value function V*(x)", "Bellman value function V*(s)"],
          ["HJB equation (PDE)", "Bellman optimality equation"],
          ["Riccati equation (DARE)", "Critic training (MSE regression)"],
          ["Optimal gain K* x (linear)", "Policy π_θ (neural network)"],
          ["Pontryagin maximum principle", "Policy gradient theorem"],
          ["Analytic solution (if available)", "Sampled solution (always works)"],
        ].map(([ctrl, rl], i) => (
          <>
            <div key={`c${i}`} style={{ background: "var(--color-background-primary)", padding: "8px 14px", fontSize: 12.5, color: "var(--color-text-primary)", borderTop: "0.5px solid var(--color-border-tertiary)" }}>{ctrl}</div>
            <div key={`r${i}`} style={{ background: "var(--color-background-primary)", padding: "8px 14px", fontSize: 12.5, color: "var(--color-text-primary)", borderTop: "0.5px solid var(--color-border-tertiary)" }}>{rl}</div>
          </>
        ))}
      </div>

      <Callout type="result" title="Experiment 4: Critic converges to V*">
        Training on 2D LQR, comparing V_φ(s) against analytic V*(s) = −sᵀP*s:
        <ul style={{ paddingLeft: 16, margin: "8px 0" }}>
          <li>Iteration 10: correlation ρ = 0.34 (critic is mostly random)</li>
          <li>Iteration 100: ρ = 0.88 (good shape, wrong scale)</li>
          <li>Iteration 250: ρ = 0.98 (essentially recovered V*)</li>
        </ul>
        The neural network critic, trained purely from reward signals with no knowledge of A, B, Q, R, converges to the same quadratic bowl that Riccati derived analytically.
      </Callout>

      <Callout type="insight" title="The punchline of the entire project">
        The Bellman equation <em>is</em> the HJB equation in discrete time. RL and optimal control are two languages for the same mathematical object. The difference is the solution method: control theory solves it analytically (when possible); RL approximates it with neural networks (always possible). The ρ=0.98 result means that after 250 iterations of pure reward-signal learning, the RL agent has essentially recovered Riccati's 1960 result.
      </Callout>

      <Callout type="thought" title="What does this imply about neural networks and physics?">
        The critic is an unstructured MLP — no physics built in, no knowledge of matrix equations. Yet it converges to a quadratic form (V*=−xᵀP*x) because the Bellman equation forces it to. What does this say about the inductive bias of neural networks trained with RL? Could you design a network architecture that converges faster by baking in the quadratic structure? (This is the field of <em>physics-informed neural networks</em>.)
      </Callout>
    </div>
  );
}

function Ch09() {
  return (
    <div>
      <Tag color="#10b981">Chapter 9</Tag>
      <H1>Codebase Architecture</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>How the project was designed — patterns, structure, decisions, and trade-offs.</em></P>
      <Divider />

      <H2>Directory Structure & Philosophy</H2>
      <div style={{ fontFamily: "ui-monospace,monospace", fontSize: 12.5, background: "#1e1e2e", color: "#cdd6f4", padding: "1.2rem 1.4rem", borderRadius: 10, margin: "1rem 0", lineHeight: 2 }}>
        <div style={{ color: "#89b4fa" }}>pg_project/</div>
        <div style={{ paddingLeft: 20 }}>
          <div style={{ color: "#a6e3a1" }}>src/</div>
          <div style={{ paddingLeft: 20 }}>
            <div><span style={{ color: "#89b4fa" }}>envs/</span><span style={{ color: "#6c7086" }}>  ← everything the agent interacts with</span></div>
            <div style={{ paddingLeft: 20, color: "#cdd6f4" }}>base.py, lqr.py, continuous_envs.py</div>
            <div><span style={{ color: "#89b4fa" }}>agents/</span><span style={{ color: "#6c7086" }}>← everything that decides and learns</span></div>
            <div style={{ paddingLeft: 20, color: "#cdd6f4" }}>policy.py, value.py, reinforce.py, npg.py, baselines.py</div>
            <div><span style={{ color: "#89b4fa" }}>utils/</span><span style={{ color: "#6c7086" }}>  ← evaluation, plotting, analysis</span></div>
            <div style={{ paddingLeft: 20, color: "#cdd6f4" }}>utils.py, variance_analysis.py, hjb_analysis.py</div>
          </div>
          <div style={{ color: "#f38ba8" }}>experiments/</div>
          <div style={{ paddingLeft: 20, color: "#cdd6f4" }}>exp1_lqr_baseline.py, exp2_continuous_control.py, ...</div>
          <div style={{ color: "#cba6f7" }}>report/</div>
          <div style={{ paddingLeft: 20, color: "#cdd6f4" }}>main.tex, references.bib</div>
          <div style={{ color: "#fab387" }}>run_all_experiments.py</div>
        </div>
      </div>

      <P>The core design principle: <strong>separation of concerns</strong>. Environments know nothing about agents. Agents know nothing about experiments. Utilities are stateless functions. This makes every piece individually testable and swappable.</P>

      <H2>Key Design Patterns Used</H2>

      <H3>1. Abstract Base Class (environments)</H3>
      <P>MDPEnv defines the contract. All environments implement it. All agents are written against MDPEnv. This means you can swap LQREnv for ContinuousPendulum with a single line change in an experiment script — nothing else changes.</P>

      <Callout type="arch" title="Why ABCs over duck typing?">
        Python's duck typing would let you use any object with a <IL>step()</IL> method. But ABCs give you: (1) explicit error messages ("forgot to implement reset()") at instantiation time rather than runtime, (2) IDE autocomplete because the interface is documented, (3) clear documentation of what the contract is. For a research codebase that you'll come back to in 3 months, this pays off.
      </Callout>

      <H3>2. Dependency injection (agents)</H3>
      <P>Agents receive their policy and critic as constructor arguments — they don't create them internally. This is Dependency Injection. Look at how experiments create agents:</P>
      <CodeBlock code={C.expArch} />
      <P>This means you can: (a) test the policy separately from the agent, (b) share a policy between agents, (c) swap policy architectures without touching agent code.</P>

      <H3>3. Factory functions (experiments)</H3>
      <P>The <IL>make_policy()</IL> factory ensures every agent gets a freshly initialised policy with the same architecture. Without this, two agents sharing a policy object would interfere — gradient updates from one would affect the other.</P>

      <H3>4. Trajectory container (data layer)</H3>
      <P>The <IL>Trajectory</IL> class (in reinforce.py) holds a single episode's data and computes returns. This clean separation between data collection and learning makes the code easy to test: you can construct a Trajectory by hand and check that returns are computed correctly before training anything.</P>

      <Callout type="arch" title="Why not just use lists?">
        You could store states/actions/rewards in plain lists. But a <IL>Trajectory</IL> class centralises the return computation logic (including the backward discount loop), makes the data self-describing, and lets you add methods (like <IL>compute_gae()</IL>) without polluting the agent code. Research code that scales well tends to have clean data containers.
      </Callout>

      <H3>5. Pure functions (utils)</H3>
      <P>All utility functions in <IL>src/utils/</IL> are stateless — they take inputs and return outputs, no side effects except for saving files. This makes them trivially testable and composable. The plotting functions can be called from any experiment without worrying about global state.</P>

      <H2>The Training Loop Pattern</H2>
      <P>Every agent's <IL>train()</IL> method follows the same loop:</P>
      <Math>{"for iteration in range(n_iterations):\n    1. collect_batch(env, n_episodes)   <- data collection\n    2. update(trajectories)              <- gradient update + return metrics\n    3. log metrics to history            <- for plotting later\n    4. optionally print progress"}</Math>
      <P>The loop is the same across all three algorithms. Only <IL>update()</IL> differs. This pattern makes comparing algorithms easy and ensures apples-to-apples experimental comparisons.</P>

      <Callout type="thought" title="How would you add PPO to this codebase?">
        PPO (Proximal Policy Optimisation) clips the policy gradient to prevent large updates. Where would you add it? What would you need to modify? The answer is: only the <IL>update()</IL> method in a new <IL>PPOAgent</IL> class. The collect loop, trajectory container, environment interface, and plotting utilities would all be reused unchanged. This is the payoff of the architectural decisions above.
      </Callout>
    </div>
  );
}

function Ch10() {
  return (
    <div>
      <Tag color="#ef4444">Chapter 10</Tag>
      <H1>Deep Questions</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>The questions that separate people who've read about RL from people who understand it.</em></P>
      <Divider />

      <H2>Tricky Questions (with answers)</H2>

      {[
        {
          q: "REINFORCE is an unbiased estimator. If it's unbiased, why does it converge to a suboptimal policy?",
          a: "Unbiased means E[ĝ] = ∇J(θ) — the gradient direction is correct on average. But with high variance, the actual gradient sample ĝ points in a wildly wrong direction most of the time. Gradient ascent with a noisy gradient still converges, but slowly and to wherever the noise pushes it. Additionally, unbiasedness is about gradient estimation — the optimal gradient direction doesn't guarantee convergence to the global optimum if J(θ) is non-convex (which it always is for neural network policies). REINFORCE can converge to local optima.",
        },
        {
          q: "In the LQR experiment, the learned policy gets within 2.3% of the analytic optimum. What explains the remaining 2.3%?",
          a: "Several factors: (1) The neural network can represent any policy, including K*x, but gradient ascent doesn't guarantee finding the global optimum. (2) The discount factor γ=0.99 means the RL objective is slightly different from the infinite-horizon LQR cost. (3) Process noise means even the optimal policy doesn't achieve the noiseless return. (4) Finite episodes — the episode ends at 200 steps, while DARE assumes infinite horizon. (5) Function approximation error in the critic propagates into advantage estimates.",
        },
        {
          q: "The GAE formula uses V(s_{t+1}) for the final step, but the episode is done. What value should V(s_{T+1}) take?",
          a: "Zero. When an episode terminates (done=True), there is no future reward — V(s_{T+1}) = 0 by definition. In the code, this is handled by the mask: delta = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t). When done=True, the mask is 0 and the bootstrap term vanishes. Getting this wrong is a subtle bug that causes a systematic bias in the advantage estimate, particularly for episodes that end due to task completion vs timeout.",
        },
        {
          q: "The natural gradient uses F^{-1}g. What happens if F is singular (non-invertible)?",
          a: "F is always positive semi-definite (as a covariance matrix of scores), so it can be singular. Singularity occurs when some policy parameters don't affect the distribution — redundant parameterisations. The Tikhonov damping term (F + εI) added in the CG step ensures we solve (F+εI)x = g instead, which is always invertible. The ε=0.01 damping used in this project means the natural gradient is actually F_damped^{-1}g — a regularised version that's numerically stable.",
        },
        {
          q: "If you use a really bad critic (one that always predicts V(s)=0), what happens to the REINFORCE update?",
          a: "V(s)=0 everywhere means advantages equal the raw returns G. So A2C degrades to REINFORCE with no baseline — high variance, slow convergence. But importantly, it's still unbiased! E[∇log π · (G - 0)] = E[∇log π · G] = ∇J. The bias only enters when the critic is systematically wrong in a correlated-with-gradient way (e.g., overestimates returns in states where good actions are taken). A zero critic is bad but unbiased. A biased critic — like one trained on old data from a different policy — can be much worse.",
        },
      ].map((qa, i) => (
        <div key={i} style={{ margin: "1.5rem 0" }}>
          <Callout type="tricky" title={`Question ${i+1}`}>
            <p style={{ fontWeight: 600, marginBottom: 12, fontSize: 14 }}>{qa.q}</p>
            <p style={{ borderTop: "1px solid rgba(239,68,68,0.2)", paddingTop: 10, lineHeight: 1.75 }}><strong>Answer:</strong> {qa.a}</p>
          </Callout>
        </div>
      ))}

      <H2>Thought-Provoking Open Questions</H2>

      {[
        {
          q: "If you ran all four experiments 100 times with different random seeds, what would the distribution of results look like? Would A2C always beat REINFORCE?",
          insight: "RL results are famously variable across seeds — small random differences in early trajectories can lead to very different local optima. Papers often cherry-pick good seeds. Running 100 seeds would show: (1) significant overlap in distributions between algorithms, (2) some seeds where REINFORCE outperforms A2C, (3) the mean performance ordering holds but individual trials are noisy. This is why reporting mean ± std over multiple seeds (as in Experiment 3) is essential for honest science."
        },
        {
          q: "The critic converges to V*(s) = −sᵀP*s with ρ=0.98 correlation. But neural networks can represent any function, not just quadratics. What is it about the training signal that forces the quadratic structure?",
          insight: "The Bellman equation V(s) = E[r + γV(s')] is the constraint. For LQR with linear dynamics and quadratic reward, the only function that satisfies Bellman consistency everywhere is the quadratic V*(s) = −sᵀP*s. Any other function shape would violate the equation at some states, generating a non-zero MSE loss that gradient descent corrects. The Bellman equation effectively encodes the physics of the system into the critic's training signal — the NN architecture doesn't matter, only the fixed point of the equation does."
        },
        {
          q: "This project learns a policy without knowing system dynamics. But in robotics, data is expensive — each episode requires physical robot operation. How would you adapt these methods to the 10-episode regime?",
          insight: "This is the core challenge of real-world RL: sample efficiency. Options include: (1) Model-based RL — learn a dynamics model from few episodes, simulate many more episodes, (2) Gaussian Processes for dynamics uncertainty — plan under uncertainty, (3) Meta-learning — train on many similar tasks so you need few episodes for a new one, (4) Transfer from simulation with domain randomisation. The methods in this project are fundamentally sample-inefficient for real systems but provide the theoretical foundation for understanding why sample-efficient methods work."
        },
      ].map((qa, i) => (
        <div key={i} style={{ margin: "1.5rem 0" }}>
          <Callout type="thought" title={`Open Question ${i+1}`}>
            <p style={{ fontWeight: 600, marginBottom: 12, fontSize: 14 }}>{qa.q}</p>
            <p style={{ borderTop: "1px solid rgba(139,92,246,0.2)", paddingTop: 10, lineHeight: 1.75, fontStyle: "italic" }}><strong>Discussion:</strong> {qa.insight}</p>
          </Callout>
        </div>
      ))}
    </div>
  );
}

function Ch11() {
  const refs = [
    { key: "Williams1992", cite: "Williams, R.J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256.", note: "The original REINFORCE paper. Introduces the log-derivative trick and the score function gradient estimator. Essential reading." },
    { key: "Sutton1999", cite: "Sutton, R.S., McAllester, D., Singh, S., Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. NeurIPS 12.", note: "Proves the policy gradient theorem in its modern form. Shows that the gradient can be estimated with function approximation without knowing the state distribution." },
    { key: "Kakade2001", cite: "Kakade, S. (2001). A natural policy gradient. NeurIPS 14.", note: "Introduces the natural policy gradient using the Fisher Information Matrix. Shows it corresponds to steepest ascent in distribution space." },
    { key: "Schulman2015a", cite: "Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. ICLR 2016.", note: "Introduces GAE (λ-weighted TD errors). Derives the bias-variance trade-off precisely and shows λ=0.95–0.99 works well in practice." },
    { key: "Schulman2015b", cite: "Schulman, J., Levine, S., Abbeel, P., Jordan, M., Moritz, P. (2015). Trust region policy optimization. ICML.", note: "Builds on NPG with a trust-region constraint. Shows monotonic improvement guarantees. The CG implementation in this project is directly from TRPO." },
    { key: "Mnih2016", cite: "Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.", note: "Introduces A3C (asynchronous A2C). Demonstrates that the actor-critic framework scales to complex Atari and 3D games." },
    { key: "SuttonBarto2018", cite: "Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.", note: "The standard textbook. Chapters 13 (policy gradient methods) and 9 (function approximation) are the theoretical backbone of this project." },
    { key: "Bertsekas2019", cite: "Bertsekas, D. (2019). Reinforcement Learning and Optimal Control. Athena Scientific.", note: "Covers the connection between RL and dynamic programming / optimal control. Chapter 2 is essential for understanding the HJB connection." },
    { key: "Anderson2007", cite: "Anderson, B.D.O. & Moore, J.B. (2007). Optimal Control: Linear Quadratic Methods. Dover Publications.", note: "The definitive reference for LQR theory. Covers the Riccati equation, DARE, stability analysis, and optimal gain derivation." },
    { key: "Paszke2019", cite: "Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.", note: "The automatic differentiation library used throughout the project. The autograd engine is what makes computing ∇log π(a|s) trivial." },
  ];

  return (
    <div>
      <Tag color="#6c7086">Chapter 11</Tag>
      <H1>Bibliography</H1>
      <P><em style={{ color: "var(--color-text-secondary)", fontSize: 14 }}>All references used in this project, with reading notes.</em></P>
      <Divider />
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {refs.map(r => (
          <div key={r.key} style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "14px 16px" }}>
            <p style={{ fontFamily: "ui-monospace,monospace", fontSize: 11, color: "#8b5cf6", marginBottom: 6, fontWeight: 700 }}>[{r.key}]</p>
            <p style={{ fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.6, marginBottom: 6 }}>{r.cite}</p>
            <p style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.6, fontStyle: "italic" }}>{r.note}</p>
          </div>
        ))}
      </div>
      <Divider />
      <P style={{ fontSize: 12, color: "var(--color-text-secondary)", textAlign: "center" }}>
        These notes were written as a companion to the Policy Gradient Methods for Continuous Decision Processes project (July 2025). They are intended to be read alongside the source code in <IL>src/</IL>.
      </P>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAPTER REGISTRY
// ═══════════════════════════════════════════════════════════════════════════

const CHAPTERS = [
  { title: "Introduction",             icon: "📖", tag: "Ch 1",  comp: Ch01 },
  { title: "Markov Decision Processes",icon: "🔄", tag: "Ch 2",  comp: Ch02 },
  { title: "Policy Gradient Theorem",  icon: "📐", tag: "Ch 3",  comp: Ch03 },
  { title: "Value Functions",          icon: "🧠", tag: "Ch 4",  comp: Ch04 },
  { title: "Variance & Baselines",     icon: "📉", tag: "Ch 5",  comp: Ch05 },
  { title: "Gaussian Policies",        icon: "🎯", tag: "Ch 6",  comp: Ch06 },
  { title: "Three Algorithms",         icon: "⚙️",  tag: "Ch 7",  comp: Ch07 },
  { title: "HJB & Optimal Control",    icon: "🔗", tag: "Ch 8",  comp: Ch08 },
  { title: "Codebase Architecture",    icon: "🏗️", tag: "Ch 9",  comp: Ch09 },
  { title: "Deep Questions",           icon: "⚠️", tag: "Ch 10", comp: Ch10 },
  { title: "Bibliography",             icon: "📚", tag: "Ch 11", comp: Ch11 },
];

// ═══════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════

export default function ProjectNotes() {
  const [active, setActive] = useState(0);
  const Active = CHAPTERS[active].comp;

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", fontFamily: "system-ui, sans-serif", background: "var(--color-background-primary)" }}>

      {/* ── Sidebar ── */}
      <aside style={{ width: 240, flexShrink: 0, background: "#11111b", display: "flex", flexDirection: "column", overflow: "hidden", borderRight: "1px solid #313244" }}>
        <div style={{ padding: "18px 16px 14px", borderBottom: "1px solid #313244" }}>
          <p style={{ fontFamily: "ui-monospace,monospace", fontSize: 10, color: "#6c7086", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>Project Notes</p>
          <p style={{ fontSize: 14, fontWeight: 800, color: "#cdd6f4", lineHeight: 1.3 }}>Policy Gradient Methods</p>
          <p style={{ fontSize: 11, color: "#6c7086", marginTop: 4 }}>July 2025 · 11 chapters</p>
        </div>
        <nav style={{ flex: 1, overflowY: "auto", padding: "10px 8px" }}>
          {CHAPTERS.map((ch, i) => (
            <button key={i} onClick={() => setActive(i)} style={{
              width: "100%", display: "flex", alignItems: "center", gap: 10, padding: "9px 10px",
              borderRadius: 7, border: "none", cursor: "pointer", textAlign: "left", marginBottom: 2,
              background: active === i ? "#1e1e2e" : "transparent",
              transition: "background 0.15s",
            }}>
              <span style={{ fontSize: 15, flexShrink: 0 }}>{ch.icon}</span>
              <div style={{ minWidth: 0 }}>
                <p style={{ fontSize: 10, color: active === i ? "#cba6f7" : "#6c7086", fontFamily: "ui-monospace,monospace", marginBottom: 1 }}>{ch.tag}</p>
                <p style={{ fontSize: 12, color: active === i ? "#cdd6f4" : "#a6adc8", fontWeight: active === i ? 600 : 400, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{ch.title}</p>
              </div>
              {active === i && <div style={{ width: 3, height: 3, borderRadius: "50%", background: "#cba6f7", marginLeft: "auto", flexShrink: 0 }} />}
            </button>
          ))}
        </nav>
        <div style={{ padding: "12px 16px", borderTop: "1px solid #313244" }}>
          <p style={{ fontSize: 10, color: "#6c7086", lineHeight: 1.6, fontFamily: "ui-monospace,monospace" }}>
            Tap any chapter to navigate.<br />
            Code blocks have a copy button.
          </p>
        </div>
      </aside>

      {/* ── Content ── */}
      <main style={{ flex: 1, overflowY: "auto", padding: "2.2rem 2.5rem 4rem", background: "var(--color-background-primary)" }}>
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          <Active />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: "3rem", paddingTop: "1.5rem", borderTop: "1px solid var(--color-border-tertiary)" }}>
            {active > 0
              ? <button onClick={() => setActive(active - 1)} style={{ fontSize: 13, color: "var(--color-text-secondary)", background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                  ← {CHAPTERS[active - 1].title}
                </button>
              : <div />}
            {active < CHAPTERS.length - 1
              ? <button onClick={() => setActive(active + 1)} style={{ fontSize: 13, color: "var(--color-text-secondary)", background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                  {CHAPTERS[active + 1].title} →
                </button>
              : <div />}
          </div>
        </div>
      </main>
    </div>
  );
}
