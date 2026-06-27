"""
iql.py
======
Implicit Q-Learning (IQL) for discrete action spaces.

Reference
---------
Kostrikov, I., Nair, A., & Levine, S. (2022).
"Offline Reinforcement Learning with Implicit Q-Learning."
ICLR 2022. arXiv:2110.06169.

IQL has three networks:
  Q(s,a) : two-critic ensemble Q1, Q2 (Clipped Double Q, like SAC)
  V(s)   : value function trained with *expectile* regression
  π(a|s) : policy trained with advantage-weighted regression (AWR)

Three losses:
  L_V = E[ ρ_τ( Q_target(s,a) - V(s) )^2 ]
        where ρ_τ(u) = | τ - 1(u<0) |  (asymmetric expectile)
  L_Q = E[ ( r + γ V(s') - Q(s,a) )^2 ]      (no max over actions!)
  L_π = - E[ exp(β (Q(s,a) - V(s)))  ·  log π(a|s) ]      (AWR)

Crucially, IQL **never queries Q at out-of-distribution actions** —
it bootstraps with V(s'), not max_a Q(s', a). This is what makes it
safe for true offline learning.

For our discrete action space we use a softmax-policy network with
n_actions outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------- #
def mlp(in_dim: int, out_dim: int, hidden: int = 256,
        n_hidden: int = 3, activation: str = "relu") -> nn.Module:
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class QNet(nn.Module):
    """Q(s,a) for discrete actions: outputs a vector over actions."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256,
                 n_hidden: int = 3):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden, n_hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class VNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, n_hidden: int = 3):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden, n_hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PolicyNet(nn.Module):
    """Categorical policy over discrete actions."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256,
                 n_hidden: int = 3):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden, n_hidden)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).log_prob(actions)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.forward(obs)
        if deterministic:
            return dist.probs.argmax(dim=-1)
        return dist.sample()


# --------------------------------------------------------------------- #
# Expectile loss
# --------------------------------------------------------------------- #
def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Asymmetric L2 loss.  When tau=0.5 this is plain MSE.
    For tau>0.5 the loss penalises *underestimates* of the target more
    than overestimates — V(s) is pushed toward the upper expectile of
    Q(s,a) under the dataset distribution. This is the key
    OOD-avoidance mechanism of IQL.
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


# --------------------------------------------------------------------- #
# IQL agent
# --------------------------------------------------------------------- #
@dataclass
class IQLConfig:
    obs_dim: int
    n_actions: int
    tau_expectile: float = 0.80
    beta_awr: float = 3.0
    awr_weight_max: float = 100.0
    gamma: float = 0.995
    polyak_tau: float = 0.005
    hidden_dim: int = 256
    n_hidden: int = 3
    activation: str = "relu"
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_value: float = 3e-4
    grad_clip: float = 1.0
    device: str = "cpu"


class IQLAgent:
    """Discrete-action IQL."""

    def __init__(self, cfg: IQLConfig):
        self.cfg = cfg
        d = cfg.device
        # Twin Q networks + targets
        self.q1 = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim,
                       cfg.n_hidden).to(d)
        self.q2 = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim,
                       cfg.n_hidden).to(d)
        self.q1_target = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim,
                              cfg.n_hidden).to(d)
        self.q2_target = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim,
                              cfg.n_hidden).to(d)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        # Value network
        self.v = VNet(cfg.obs_dim, cfg.hidden_dim, cfg.n_hidden).to(d)
        # Policy
        self.pi = PolicyNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim,
                            cfg.n_hidden).to(d)

        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.lr_critic,
        )
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.lr_value)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_actor)

    # ----- helpers ----------------------------------------------------
    def _polyak(self):
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_t.data.mul_(1 - self.cfg.polyak_tau).add_(
                    self.cfg.polyak_tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_t.data.mul_(1 - self.cfg.polyak_tau).add_(
                    self.cfg.polyak_tau * p.data)

    # ----- update -----------------------------------------------------
    def update(self, batch: dict) -> dict:
        """
        batch keys: obs, act (long), rew, next_obs, done  (all torch tensors)
        """
        obs, act, rew, next_obs, done = (batch[k] for k in
            ("obs", "act", "rew", "next_obs", "done"))

        # --- V update: expectile regression to min(Q1,Q2)_target ---
        with torch.no_grad():
            q1_t = self.q1_target(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)
            q2_t = self.q2_target(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)
            q_min = torch.min(q1_t, q2_t)
        v_pred = self.v(obs)
        v_loss = expectile_loss(q_min - v_pred, self.cfg.tau_expectile)
        self.v_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.v.parameters(), self.cfg.grad_clip)
        self.v_opt.step()

        # --- Q update: bootstrap with V(s'), NOT max Q ---
        with torch.no_grad():
            target = rew + self.cfg.gamma * (1 - done) * self.v(next_obs)
        q1_pred = self.q1(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)
        q2_pred = self.q2(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.grad_clip,
        )
        self.q_opt.step()

        self._polyak()

        # --- Policy update: AWR with per-batch advantage normalisation ---
        # Growing Q-values (expected with gamma=0.995 over 4320-step episodes)
        # cause raw exp(beta*adv) to saturate the clamp on most samples, making
        # the policy gradient noisy.  Normalising adv to zero-mean / unit-std
        # keeps the effective beta scale stable throughout training.
        with torch.no_grad():
            q1_t2 = self.q1_target(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)
            q2_t2 = self.q2_target(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)
            q_min2 = torch.min(q1_t2, q2_t2)
            adv = q_min2 - self.v(obs).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalise
            w = torch.exp(self.cfg.beta_awr * adv).clamp(
                max=self.cfg.awr_weight_max)
        log_p = self.pi.log_prob(obs, act)
        pi_loss = -(w * log_p).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.grad_clip)
        self.pi_opt.step()

        return {
            "loss/v": float(v_loss.item()),
            "loss/q": float(q_loss.item()),
            "loss/pi": float(pi_loss.item()),
            "stat/q_mean": float(q_min.mean().item()),
            "stat/v_mean": float(v_pred.mean().item()),
            "stat/adv_mean": float(adv.mean().item()),
            "stat/awr_w_mean": float(w.mean().item()),
        }

    # ----- inference --------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.cfg.device).unsqueeze(0)
        a = self.pi.act(obs_t, deterministic=deterministic)
        return int(a.item())

    # ----- persistence ------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "v":  self.v.state_dict(),  "pi": self.pi.state_dict(),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, sd: dict):
        self.q1.load_state_dict(sd["q1"]); self.q2.load_state_dict(sd["q2"])
        self.q1_target.load_state_dict(sd["q1_target"])
        self.q2_target.load_state_dict(sd["q2_target"])
        self.v.load_state_dict(sd["v"]); self.pi.load_state_dict(sd["pi"])


# --------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed-size circular buffer for (s,a,r,s',done) transitions."""

    def __init__(self, capacity: int, obs_dim: int):
        self.cap = capacity
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros((capacity,), dtype=np.int64)
        self.rew  = np.zeros((capacity,), dtype=np.float32)
        self.nxt  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, n, d):
        self.obs[self.ptr]  = o
        self.act[self.ptr]  = a
        self.rew[self.ptr]  = r
        self.nxt[self.ptr]  = n
        self.done[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, device: str = "cpu") -> dict:
        idx = np.random.randint(0, self.size, size=n)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "act": torch.as_tensor(self.act[idx], device=device),
            "rew": torch.as_tensor(self.rew[idx], device=device),
            "next_obs": torch.as_tensor(self.nxt[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }
