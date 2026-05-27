"""
SAC-Discrete training for the OuterRimEnv (4-direction grid exploration).

Standard SAC (stable_baselines3.SAC) only supports continuous action spaces,
so this implementation follows the SAC-Discrete variant
(Christodoulou, 2019: https://arxiv.org/abs/1910.07207):

  - Actor outputs a categorical distribution over the 4 discrete actions.
  - Twin Q-networks output Q-values for all actions; expectations are computed
    analytically as sum_a pi(a|s) * Q(s,a) instead of via reparameterization.
  - Temperature alpha is tuned automatically against a target entropy.

Run:
    python train_sac.py
"""

import os
# Force pygame to use a dummy video driver during training so a window does
# not pop up for every env.reset() in headless contexts.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict as TDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from main import OuterRimEnv  # noqa: E402  (env class lives in main.py)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SACConfig:
    total_timesteps: int = 150_000
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005                # Polyak soft-update coefficient
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    hidden_dim: int = 256
    train_freq: int = 1               # env steps between gradient updates
    gradient_steps: int = 1           # gradient updates per call
    target_update_interval: int = 1   # gradient steps between target sync
    target_entropy_ratio: float = 0.98  # target_ent = ratio * log|A|  (max ent)
    log_interval_steps: int = 1_000
    checkpoint_every: int = 25_000
    seed: int = 0


CFG = SACConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Observation handling
# ---------------------------------------------------------------------------
def obs_to_vec(obs: TDict[str, np.ndarray]) -> np.ndarray:
    """Flatten the Dict observation into a single float vector in [0, 1]."""
    vision = obs["vision"].astype(np.float32).reshape(-1) / 3.0
    memory = obs["seen_memory"].astype(np.float32).reshape(-1)
    return np.concatenate([vision, memory], axis=0)


def obs_dim_for(env: OuterRimEnv) -> int:
    sample = env.observation_space.sample()
    return obs_to_vec(sample).shape[0]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CategoricalActor(nn.Module):
    """Outputs logits over discrete actions; returns probs & log-probs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.body = MLP(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.body(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        probs, _ = self.forward(obs)
        if deterministic:
            return int(torch.argmax(probs, dim=-1).item())
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())


class TwinQ(nn.Module):
    """Two Q-networks that each output Q-values for every discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.q1 = MLP(obs_dim, n_actions, hidden)
        self.q2 = MLP(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs), self.q2(obs)


# ---------------------------------------------------------------------------
# Replay buffer (numpy, fixed capacity, circular)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=DEVICE),
            torch.as_tensor(self.actions[idx], device=DEVICE),
            torch.as_tensor(self.rewards[idx], device=DEVICE),
            torch.as_tensor(self.next_obs[idx], device=DEVICE),
            torch.as_tensor(self.dones[idx], device=DEVICE),
        )


# ---------------------------------------------------------------------------
# SAC-Discrete agent
# ---------------------------------------------------------------------------
class SACDiscrete:
    def __init__(self, obs_dim: int, n_actions: int, cfg: SACConfig):
        self.cfg = cfg
        self.n_actions = n_actions

        self.actor = CategoricalActor(obs_dim, n_actions, cfg.hidden_dim).to(DEVICE)
        self.critic = TwinQ(obs_dim, n_actions, cfg.hidden_dim).to(DEVICE)
        self.critic_target = TwinQ(obs_dim, n_actions, cfg.hidden_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr_critic)

        # Automatic temperature tuning.  target entropy = ratio * log(|A|)
        self.target_entropy = cfg.target_entropy_ratio * np.log(n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs_vec: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return self.actor.act(obs_t, deterministic=deterministic)

    def update(self, batch) -> TDict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        # ----- Critic update ----------------------------------------------
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs)
            next_q1_t, next_q2_t = self.critic_target(next_obs)
            min_next_q = torch.min(next_q1_t, next_q2_t)
            # V(s') = sum_a pi(a|s') * (Q_target(s',a) - alpha * log pi(a|s'))
            v_next = (next_probs * (min_next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * v_next

        q1_all, q2_all = self.critic(obs)
        a_idx = actions.unsqueeze(-1)
        q1 = q1_all.gather(1, a_idx).squeeze(-1)
        q2 = q2_all.gather(1, a_idx).squeeze(-1)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----- Actor update -----------------------------------------------
        probs, log_probs = self.actor(obs)
        with torch.no_grad():
            q1_all_a, q2_all_a = self.critic(obs)
            min_q = torch.min(q1_all_a, q2_all_a)
        # E_a~pi [ alpha * log pi(a|s) - Q(s,a) ]
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----- Temperature update -----------------------------------------
        # entropy = -sum_a pi(a|s) * log pi(a|s)
        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(entropy.mean().item()),
            "q1_mean": float(q1.mean().item()),
        }

    def soft_update_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1.0 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(DEVICE))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: SACConfig = CFG):
    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    save_dir = os.path.join("Training_Star_Wars_Galaxy_Phase_1", "Saved RL Models")
    log_dir = os.path.join("Training_Star_Wars_Galaxy_Phase_1", "logs", "SAC")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, "SAC_Discrete_Model_Star_Wars_Galaxy")

    env = OuterRimEnv()
    n_actions = env.action_space.n
    obs_dim = obs_dim_for(env)
    print(f"[SAC-Discrete] obs_dim={obs_dim}  n_actions={n_actions}  device={DEVICE}")

    agent = SACDiscrete(obs_dim, n_actions, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim)

    obs_dict, _ = env.reset()
    obs_vec = obs_to_vec(obs_dict)

    ep_return = 0.0
    ep_len = 0
    ep_returns = deque(maxlen=20)
    ep_enemies = deque(maxlen=20)
    ep_coverage = deque(maxlen=20)
    last_log_time = time.time()

    metrics: TDict[str, float] = {}

    for step in range(1, cfg.total_timesteps + 1):
        # ----- act -------------------------------------------------------
        if step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs_vec)

        next_obs_dict, reward, done, truncated, info = env.step(action)
        next_obs_vec = obs_to_vec(next_obs_dict)
        # Bootstrap on time-limit truncations rather than treating them as terminal.
        store_done = done and not truncated
        buffer.add(obs_vec, action, reward, next_obs_vec, store_done)

        obs_vec = next_obs_vec
        ep_return += reward
        ep_len += 1

        if done or truncated:
            coverage = float(np.mean(env.seen_map)) * 100.0
            ep_returns.append(ep_return)
            ep_enemies.append(info.get("visited_enemies", 0))
            ep_coverage.append(coverage)
            obs_dict, _ = env.reset()
            obs_vec = obs_to_vec(obs_dict)
            ep_return = 0.0
            ep_len = 0

        # ----- learn -----------------------------------------------------
        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                batch = buffer.sample(cfg.batch_size)
                metrics = agent.update(batch)
                if step % cfg.target_update_interval == 0:
                    agent.soft_update_target()

        # ----- logging ---------------------------------------------------
        if step % cfg.log_interval_steps == 0:
            now = time.time()
            sps = cfg.log_interval_steps / max(1e-6, now - last_log_time)
            last_log_time = now
            mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
            mean_enemies = np.mean(ep_enemies) if ep_enemies else float("nan")
            mean_cov = np.mean(ep_coverage) if ep_coverage else float("nan")
            print(
                f"step={step:7d} | sps={sps:5.0f} | "
                f"ret(mean,20)={mean_ret:7.2f} | "
                f"enemies(mean,20)={mean_enemies:5.2f} | "
                f"coverage(mean,20)={mean_cov:5.1f}% | "
                f"alpha={metrics.get('alpha', float('nan')):.3f} | "
                f"entropy={metrics.get('entropy', float('nan')):.3f}"
            )

        if step % cfg.checkpoint_every == 0:
            ckpt_path = f"{final_model_path}_{step}_steps.pt"
            agent.save(ckpt_path)
            print(f"  [checkpoint] saved {ckpt_path}")

    agent.save(final_model_path + ".pt")
    print(f"[SAC-Discrete] final model saved to {final_model_path}.pt")
    env.close() if hasattr(env, "close") else None
    return agent


if __name__ == "__main__":
    train(CFG)
