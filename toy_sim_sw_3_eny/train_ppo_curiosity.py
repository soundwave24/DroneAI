"""
train_ppo_curiosity.py  (toy_sim_sw_3_eny — dynamic entity environment)
========================================================================

Train a PPO agent equipped with an Intrinsic Curiosity Module (ICM) on the
`OuterRimEnv` Phase 2 environment (mobile entities).

Why ICM?
--------
In Phase 2 the map is 40x40 with a 15x15 vision patch, mobile Separatist
ships introduce stochastic danger, and the reward_memory halo provides a
dense signal. Nevertheless, exploration of unseen corners is still hard for a
naive policy. The ICM supplies a *self-generated* intrinsic reward that is
large in states whose dynamics the ICM cannot yet predict, ensuring the agent
explores all parts of the map.

ICM is made of three small networks:
    phi(s)         : feature encoder
    inverse model  : predicts a_t from (phi(s_t), phi(s_{t+1}))
    forward model  : predicts phi(s_{t+1}) from (phi(s_t), a_t)

Intrinsic reward = 0.5 * || forward(phi(s),a) - phi(s') ||^2 .

Phase 2 observation space (vision_radius=7 → 15x15 patches):
    "vision"        : Box(0, 4, (15, 15), uint8)
    "danger"        : Box(0.0, 1.0, (15, 15), float32)
    "reward_memory" : Box(0.0, inf, (15, 15), float32)

Key differences from Phase 1 (toy_sim_sw_2):
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - ICMNetwork: input is now 3 × (15×15) features instead of 2 × (5×5)
      * vision / 4.0  normalised
      * danger        (already 0..1)
      * reward_memory tanh-normalised (/ 10.0)
    - All paths updated to 3_eny/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from environment import OuterRimEnv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT     = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR     = os.path.join(PROJECT_ROOT, "3_eny")
LOG_DIR          = os.path.join(TRAINING_DIR, "logs_curiosity")
CHECKPOINT_DIR   = os.path.join(TRAINING_DIR, "Saved RL Models", "PPO_Curiosity_checkpoints")
FINAL_MODEL_PATH = os.path.join(TRAINING_DIR, "Saved RL Models", "PPO_Curiosity_Model_Star_Wars_Galaxy")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# PPO training
# 백만번 돌리면 너무 오래걸려서 일단 20만번으로 줄임. 나중에 필요하면 더 늘리기 TODO
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQ = 50_000
PPO_N_STEPS     = 2048
PPO_BATCH       = 512
PPO_EPOCHS      = 10

# ICM
VISION_SIZE     = 15      # 2*vision_radius+1 = 15
FEATURE_DIM     = 256     # dimensionality of phi(s) — larger for 15x15 input
ICM_LR          = 1e-3
ICM_BETA        = 0.2     # forward-loss weight (inverse weight = 1 - beta)
ICM_INTRINSIC   = 0.05    # scaling applied to intrinsic reward
ICM_TRAIN_FREQ  = PPO_N_STEPS   # train ICM once per PPO rollout
ICM_BATCH       = 256
ICM_EPOCHS      = 4

# reward_memory normalisation
REWARD_MEMORY_SCALE = 10.0

# Number of ships in the environment
N_SEP = 2
N_REP = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ICM network: encoder + inverse + forward
# ---------------------------------------------------------------------------
class ICMNetwork(nn.Module):
    """Feature encoder phi(s) plus inverse/forward dynamics heads.

    Phase 2 observation is the dict {
        "vision"        : (15,15) uint8  in [0,4],
        "danger"        : (15,15) f32    in [0,1],
        "reward_memory" : (15,15) f32    in [0,inf)
    }.
    All three fields are normalised, flattened, concatenated, and fed into the
    MLP encoder. The encoder output (FEATURE_DIM) feeds both heads.
    """

    def __init__(self, vision_size: int = VISION_SIZE, n_actions: int = 4,
                 feature_dim: int = FEATURE_DIM, hidden: int = 512,
                 reward_memory_scale: float = REWARD_MEMORY_SCALE):
        super().__init__()
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.reward_memory_scale = reward_memory_scale

        flat = vision_size * vision_size   # 225 per channel
        # 3 channels: vision (norm) + danger + reward_memory (tanh-norm)
        in_dim = 3 * flat

        # phi(s)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim), nn.ReLU(),
        )

        # Inverse model: (phi(s), phi(s')) -> action logits
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        # Forward model: (phi(s), one_hot(a)) -> phi(s')
        self.forward_head = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def _flatten_obs(self, obs: dict) -> torch.Tensor:
        """obs tensors have shape (B, 15, 15)."""
        vision = obs["vision"].float().flatten(1) / 4.0   # 0..4 -> 0..1
        danger = obs["danger"].float().flatten(1)          # already 0..1
        reward_mem = torch.tanh(
            obs["reward_memory"].float().flatten(1) / self.reward_memory_scale
        )
        return torch.cat([vision, danger, reward_mem], dim=-1)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(self._flatten_obs(obs))

    def forward(self, obs: dict, next_obs: dict, action: torch.Tensor):
        phi_s = self.encode(obs)
        phi_s_next = self.encode(next_obs)

        inv_in = torch.cat([phi_s, phi_s_next], dim=-1)
        inv_logits = self.inverse_head(inv_in)

        action_oh = F.one_hot(action.long(), num_classes=self.n_actions).float()
        fwd_in = torch.cat([phi_s, action_oh], dim=-1)
        pred_phi_next = self.forward_head(fwd_in)

        return phi_s, phi_s_next, inv_logits, pred_phi_next


# ---------------------------------------------------------------------------
# VecEnv wrapper: injects ICM intrinsic reward and stores transitions
# ---------------------------------------------------------------------------
class ICMRewardWrapper(VecEnvWrapper):
    """Adds (intrinsic_coef * intrinsic) to extrinsic reward per env step.

    Also collects the last (obs, next_obs, action) batch into `transitions`,
    which `ICMTrainingCallback` consumes when fitting the ICM network.
    """

    def __init__(self, venv, icm: ICMNetwork, device: torch.device,
                 intrinsic_coef: float = ICM_INTRINSIC):
        super().__init__(venv)
        self.icm = icm
        self.device = device
        self.intrinsic_coef = intrinsic_coef

        self._last_obs = None
        self._last_actions = None
        self.transitions: list = []  # filled by step_wait; cleared by callback

    def _obs_to_tensor(self, obs: dict) -> dict:
        return {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = obs
        return obs

    def step_async(self, actions):
        self._last_actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        next_obs, rewards, dones, infos = self.venv.step_wait()

        # --- compute intrinsic reward (no grad) -----------------------
        with torch.no_grad():
            obs_t      = self._obs_to_tensor(self._last_obs)
            next_obs_t = self._obs_to_tensor(next_obs)
            actions_t  = torch.as_tensor(self._last_actions, device=self.device)
            _, phi_s_next, _, pred_phi_next = self.icm(obs_t, next_obs_t, actions_t)
            intrinsic = 0.5 * (pred_phi_next - phi_s_next).pow(2).mean(dim=-1)
            intrinsic_np = intrinsic.detach().cpu().numpy()

        # --- store transition for ICM training ------------------------
        self.transitions.append({
            "obs":      {k: v.copy() for k, v in self._last_obs.items()},
            "next_obs": {k: v.copy() for k, v in next_obs.items()},
            "action":   self._last_actions.copy(),
        })

        total_rewards = rewards + self.intrinsic_coef * intrinsic_np

        for i, info in enumerate(infos):
            info["intrinsic_reward"] = float(intrinsic_np[i])
            info["extrinsic_reward"] = float(rewards[i])

        self._last_obs = next_obs
        return next_obs, total_rewards, dones, infos


# ---------------------------------------------------------------------------
# Callback: fits the ICM network on the buffered transitions
# ---------------------------------------------------------------------------
class ICMTrainingCallback(BaseCallback):
    """Run SGD on the ICM network every `train_freq` env steps."""

    def __init__(self, icm: ICMNetwork, wrapper: ICMRewardWrapper,
                 optimizer: optim.Optimizer, *,
                 train_freq: int = ICM_TRAIN_FREQ,
                 batch_size: int = ICM_BATCH,
                 n_epochs:   int = ICM_EPOCHS,
                 beta:       float = ICM_BETA,
                 verbose:    int = 0):
        super().__init__(verbose)
        self.icm = icm
        self.wrapper = wrapper
        self.optimizer = optimizer
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.beta = beta
        self._steps_since_train = 0

    def _on_step(self) -> bool:
        self._steps_since_train += self.training_env.num_envs
        if self._steps_since_train >= self.train_freq:
            self._steps_since_train = 0
            self._fit_icm()
        return True

    def _fit_icm(self) -> None:
        if not self.wrapper.transitions:
            return

        # Concatenate transitions along the env-batch axis.
        all_obs_v   = np.concatenate([t["obs"]["vision"]           for t in self.wrapper.transitions], axis=0)
        all_obs_d   = np.concatenate([t["obs"]["danger"]           for t in self.wrapper.transitions], axis=0)
        all_obs_rm  = np.concatenate([t["obs"]["reward_memory"]    for t in self.wrapper.transitions], axis=0)
        all_next_v  = np.concatenate([t["next_obs"]["vision"]      for t in self.wrapper.transitions], axis=0)
        all_next_d  = np.concatenate([t["next_obs"]["danger"]      for t in self.wrapper.transitions], axis=0)
        all_next_rm = np.concatenate([t["next_obs"]["reward_memory"] for t in self.wrapper.transitions], axis=0)
        all_actions = np.concatenate([t["action"]                  for t in self.wrapper.transitions], axis=0)

        n = all_actions.shape[0]
        device = next(self.icm.parameters()).device

        obs_v  = torch.as_tensor(all_obs_v, device=device)
        obs_d  = torch.as_tensor(all_obs_d, device=device)
        obs_rm = torch.as_tensor(all_obs_rm, device=device)
        nxt_v  = torch.as_tensor(all_next_v, device=device)
        nxt_d  = torch.as_tensor(all_next_d, device=device)
        nxt_rm = torch.as_tensor(all_next_rm, device=device)
        acts   = torch.as_tensor(all_actions, device=device).long()

        inv_losses, fwd_losses = [], []
        for _ in range(self.n_epochs):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                obs = {"vision": obs_v[b], "danger": obs_d[b], "reward_memory": obs_rm[b]}
                nxt = {"vision": nxt_v[b], "danger": nxt_d[b], "reward_memory": nxt_rm[b]}
                _, phi_s_next, inv_logits, pred_phi_next = self.icm(obs, nxt, acts[b])

                inv_loss = F.cross_entropy(inv_logits, acts[b])
                # Detach phi(s') so that the forward loss does not collapse the
                # encoder onto a trivial constant.
                fwd_loss = 0.5 * (pred_phi_next - phi_s_next.detach()).pow(2).mean()
                loss = (1.0 - self.beta) * inv_loss + self.beta * fwd_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                inv_losses.append(inv_loss.item())
                fwd_losses.append(fwd_loss.item())

        mean_inv = float(np.mean(inv_losses))
        mean_fwd = float(np.mean(fwd_losses))
        if self.verbose:
            print(f"[ICM] inv_loss={mean_inv:.4f}  fwd_loss={mean_fwd:.4f}  samples={n}")
        self.logger.record("icm/inverse_loss", mean_inv)
        self.logger.record("icm/forward_loss", mean_fwd)
        self.logger.record("icm/buffer_size",   n)

        # Drop transitions so the next fit only sees fresh data.
        self.wrapper.transitions = []


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def make_env():
    return OuterRimEnv(N_SEP, N_REP)


def main() -> None:
    # 1) Vectorised env
    base_env = DummyVecEnv([make_env])

    # 2) ICM network + optimizer
    icm = ICMNetwork(vision_size=VISION_SIZE, n_actions=4,
                     feature_dim=FEATURE_DIM).to(DEVICE)
    icm_optim = optim.Adam(icm.parameters(), lr=ICM_LR)

    # 3) Inject intrinsic reward
    env = ICMRewardWrapper(base_env, icm=icm, device=DEVICE)

    # 4) PPO model (MultiInputPolicy required for Dict observation space)
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH,
        n_epochs=PPO_EPOCHS,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.02,
        gae_lambda=0.95,
        vf_coef=0.5,
        gamma=0.98,
    )

    # 5) Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_curiosity",
    )
    icm_callback = ICMTrainingCallback(icm, env, icm_optim, verbose=1)
    callbacks = CallbackList([checkpoint_callback, icm_callback])

    # 6) Train
    print(f"[train_ppo_curiosity] device              : {DEVICE}")
    print(f"[train_ppo_curiosity] total_timesteps     : {TOTAL_TIMESTEPS:,}")
    print(f"[train_ppo_curiosity] tensorboard log dir : {LOG_DIR}")
    print(f"[train_ppo_curiosity] checkpoint dir      : {CHECKPOINT_DIR}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="PPO_Curiosity",
    )

    # 7) Save final PPO model and ICM weights
    model.save(FINAL_MODEL_PATH)
    torch.save(icm.state_dict(), FINAL_MODEL_PATH + "_icm.pt")
    print(f"[train_ppo_curiosity] PPO  saved to : {FINAL_MODEL_PATH}.zip")
    print(f"[train_ppo_curiosity] ICM  saved to : {FINAL_MODEL_PATH}_icm.pt")

    env.close()


if __name__ == "__main__":
    main()
