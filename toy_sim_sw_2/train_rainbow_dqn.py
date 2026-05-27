"""
Rainbow DQN training script for OuterRimEnv.

Implements the full Rainbow DQN algorithm:
    1. Double DQN          - separate target net for action evaluation
    2. Dueling architecture - separate V(s) and A(s,a) streams
    3. Prioritized Experience Replay - sample by TD-error
    4. Multi-step (n-step) returns
    5. Noisy Networks - learned parametric noise instead of epsilon-greedy
    6. Distributional RL (C51) - predict return distribution over fixed atoms

Reference:
    Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning"
    AAAI 2018, https://arxiv.org/abs/1710.02298
"""

import os
# Disable pygame display (headless training)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import math
import random
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import OuterRimEnv


# ===================================================================
# Noisy Linear layer  (Fortunato et al. 2017, factorised Gaussian noise)
# ===================================================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ===================================================================
# Rainbow Q-Network (Dueling + Noisy + Distributional)
# ===================================================================
class RainbowNet(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int,
                 n_atoms: int, v_min: float, v_max: float, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        c, h, w = obs_shape  # (2, 5, 5)

        # Small conv backbone — input is 5x5 with 2 channels (vision, seen_memory)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        flat_dim = 64 * h * w

        # Dueling streams, both noisy
        self.value_hidden = NoisyLinear(flat_dim, hidden)
        self.value_out = NoisyLinear(hidden, n_atoms)
        self.adv_hidden = NoisyLinear(flat_dim, hidden)
        self.adv_out = NoisyLinear(hidden, n_actions * n_atoms)

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over atoms for each action: shape (B, A, N)."""
        h = self.conv(x)
        v = self.value_out(F.relu(self.value_hidden(h)))                # (B, N)
        a = self.adv_out(F.relu(self.adv_hidden(h)))                    # (B, A*N)
        v = v.view(-1, 1, self.n_atoms)
        a = a.view(-1, self.n_actions, self.n_atoms)
        # Dueling: Q = V + A - mean(A)
        logits = v + a - a.mean(dim=1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        # Numerical floor so log(p) is finite
        return probs.clamp(min=1e-8)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.dist(x)                       # (B, A, N)
        return (probs * self.support).sum(dim=-1)  # (B, A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_values(x)


# ===================================================================
# Prioritized Experience Replay (proportional, sum-tree backed)
# ===================================================================
class SumTree:
    """Simple sum-tree for proportional sampling."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.size = 0
        self.write = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, data_idx: int, priority: float):
        idx = data_idx + self.capacity - 1
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float):
        data_idx = self.write
        self.update(data_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def total(self) -> float:
        return self.tree[0]

    def get(self, s: float) -> Tuple[int, float]:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int],
                 alpha: float = 0.5, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # n-step buffer
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_buffer = deque(maxlen=n_step)

    def _get_nstep(self):
        """Aggregate the deque into one n-step transition."""
        R, next_obs, done = 0.0, self.nstep_buffer[-1][3], self.nstep_buffer[-1][4]
        for i, (_, _, r, _, d) in enumerate(self.nstep_buffer):
            R += (self.gamma ** i) * r
            if d:
                # Truncate at first terminal
                next_obs = self.nstep_buffer[i][3]
                done = 1.0
                break
        return R, next_obs, done

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: float):
        self.nstep_buffer.append((obs, action, reward, next_obs, done))
        if len(self.nstep_buffer) < self.n_step:
            return
        R, n_next, n_done = self._get_nstep()
        obs0, a0, _, _, _ = self.nstep_buffer[0]

        data_idx = self.tree.add(self.max_priority ** self.alpha)
        self.obs[data_idx] = obs0
        self.actions[data_idx] = a0
        self.rewards[data_idx] = R
        self.next_obs[data_idx] = n_next
        self.dones[data_idx] = n_done

    def flush_episode(self):
        """At episode end, drain the n-step deque so partial windows still get stored."""
        while len(self.nstep_buffer) > 0:
            R, n_next, n_done = self._get_nstep()
            obs0, a0, _, _, _ = self.nstep_buffer[0]
            data_idx = self.tree.add(self.max_priority ** self.alpha)
            self.obs[data_idx] = obs0
            self.actions[data_idx] = a0
            self.rewards[data_idx] = R
            self.next_obs[data_idx] = n_next
            self.dones[data_idx] = n_done
            self.nstep_buffer.popleft()

    def __len__(self):
        return self.tree.size

    def sample(self, batch_size: int, beta: float) -> Dict[str, np.ndarray]:
        idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = random.uniform(lo, hi)
            data_idx, p = self.tree.get(s)
            idxs[i] = data_idx
            priorities[i] = p

        probs = priorities / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
            "weights": weights.astype(np.float32),
            "idxs": idxs,
        }

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(idxs, priorities):
            self.tree.update(int(idx), float(p))
            self.max_priority = max(self.max_priority, float(np.abs(td_errors).max()) + 1e-6)


# ===================================================================
# Observation packing — Dict obs -> (2, H, W) float tensor
# ===================================================================
def pack_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    # vision is 0..3, normalize to 0..1; seen_memory already 0/1
    vision = obs["vision"].astype(np.float32) / 3.0
    memory = obs["seen_memory"].astype(np.float32)
    return np.stack([vision, memory], axis=0)


# ===================================================================
# Rainbow DQN Agent
# ===================================================================
class RainbowAgent:
    def __init__(self, obs_shape, n_actions, device,
                 lr=1e-4, gamma=0.99, n_step=3,
                 v_min=-10.0, v_max=30.0, n_atoms=51,
                 buffer_size=100_000, batch_size=64,
                 target_update_freq=1000,
                 alpha=0.5, beta_start=0.4, beta_frames=200_000):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.v_min, self.v_max, self.n_atoms = v_min, v_max, n_atoms
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.online = RainbowNet(obs_shape, n_actions, n_atoms, v_min, v_max).to(device)
        self.target = RainbowNet(obs_shape, n_actions, n_atoms, v_min, v_max).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr, eps=1.5e-4)
        self.buffer = PrioritizedReplayBuffer(buffer_size, obs_shape,
                                              alpha=alpha, n_step=n_step, gamma=gamma)
        self.step_count = 0

    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.step_count / self.beta_frames)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> int:
        # Noisy nets do the exploration; no epsilon needed.
        x = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        self.online.reset_noise()
        q = self.online.q_values(x)
        return int(q.argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, float(done))

    def _projection(self, next_dist: torch.Tensor, rewards: torch.Tensor,
                    dones: torch.Tensor) -> torch.Tensor:
        """Project Bellman target T_z onto fixed atom support (Bellemare et al. C51)."""
        batch_size = rewards.size(0)
        support = self.online.support  # (N,)

        gamma_n = self.gamma ** self.n_step
        Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Edge case: when b is an integer, l == u — need to nudge so probability mass lands somewhere
        l[(u > 0) & (l == u)] -= 1
        u[(l < (self.n_atoms - 1)) & (l == u)] += 1

        m = torch.zeros(batch_size, self.n_atoms, device=self.device)
        offset = (torch.arange(batch_size, device=self.device).unsqueeze(1)
                  .expand(batch_size, self.n_atoms) * self.n_atoms)
        m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return m

    def learn(self) -> float:
        batch = self.buffer.sample(self.batch_size, self.beta())
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        weights = torch.from_numpy(batch["weights"]).to(self.device)

        # ---- target distribution (Double DQN action selection) ----
        with torch.no_grad():
            self.online.reset_noise()
            next_q_online = self.online.q_values(next_obs)
            best_actions = next_q_online.argmax(dim=1)
            self.target.reset_noise()
            next_dist_all = self.target.dist(next_obs)               # (B, A, N)
            next_dist = next_dist_all[range(self.batch_size), best_actions]  # (B, N)
            target_dist = self._projection(next_dist, rewards, dones)

        # ---- current distribution ----
        self.online.reset_noise()
        cur_dist_all = self.online.dist(obs)                         # (B, A, N)
        cur_dist = cur_dist_all[range(self.batch_size), actions]     # (B, N)

        # Cross-entropy loss per sample
        elementwise_loss = -(target_dist * cur_dist.log()).sum(dim=1)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # PER priority update — use the elementwise CE losses as |TD-error| proxy
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.buffer.update_priorities(batch["idxs"], td_errors)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.step_count = ckpt.get("step_count", 0)


# ===================================================================
# Training loop
# ===================================================================
def train(total_timesteps: int, save_path: str, log_every: int = 1000,
          learning_starts: int = 2000, train_freq: int = 4, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OuterRimEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    obs_shape = (2, 5, 5)  # (vision + seen_memory channels, H, W)
    agent = RainbowAgent(obs_shape=obs_shape, n_actions=int(env.action_space.n),
                         device=device, beta_frames=total_timesteps)

    obs_dict, _ = env.reset()
    obs = pack_obs(obs_dict)

    ep_return = 0.0
    ep_enemies = 0
    ep_len = 0
    ep_returns = deque(maxlen=20)
    ep_enemy_counts = deque(maxlen=20)
    losses = deque(maxlen=200)

    start_time = time.time()

    for t in range(1, total_timesteps + 1):
        action = agent.act(obs)
        next_obs_dict, reward, done, truncated, info = env.step(action)
        next_obs = pack_obs(next_obs_dict)

        agent.store(obs, action, reward, next_obs, done or truncated)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done or truncated:
            agent.buffer.flush_episode()
            ep_enemies = info.get("visited_enemies", 0)
            ep_returns.append(ep_return)
            ep_enemy_counts.append(ep_enemies)
            obs_dict, _ = env.reset()
            obs = pack_obs(obs_dict)
            ep_return = 0.0
            ep_len = 0

        # Train
        if len(agent.buffer) >= max(learning_starts, agent.batch_size) and t % train_freq == 0:
            loss = agent.learn()
            losses.append(loss)

        if t % log_every == 0:
            elapsed = time.time() - start_time
            fps = t / max(elapsed, 1e-6)
            mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
            mean_enemies = np.mean(ep_enemy_counts) if ep_enemy_counts else float("nan")
            mean_loss = np.mean(losses) if losses else float("nan")
            print(f"[step {t:>7d}] "
                  f"ret(20)={mean_ret:8.2f}  "
                  f"enemies(20)={mean_enemies:5.2f}  "
                  f"loss={mean_loss:6.4f}  "
                  f"beta={agent.beta():.3f}  "
                  f"buf={len(agent.buffer):>6d}  "
                  f"fps={fps:.1f}")

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\nSaved Rainbow DQN model to: {save_path}")
    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=80_000)
    p.add_argument("--train-freq", type=int, default=4,
                   help="Number of env steps between gradient updates.")
    p.add_argument("--save-path", type=str,
                   default=os.path.join("Training_Star_Wars_Galaxy_Phase_1",
                                        "Saved RL Models",
                                        "RainbowDQN_Star_Wars_Galaxy.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-starts", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(total_timesteps=args.total_timesteps,
          save_path=args.save_path,
          learning_starts=args.learning_starts,
          train_freq=args.train_freq,
          seed=args.seed)
