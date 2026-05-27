"""
train_recurrent_ppo.py  (toy_sim_sw_3_eny — dynamic entity environment)
=========================================================================

Train a Recurrent PPO (PPO + LSTM) agent on the `OuterRimEnv` environment
with mobile entities (SeperatistShip / RepublicShip).

Why a recurrent policy?
-----------------------
`OuterRimEnv` gives the agent only a 15x15 local vision patch plus danger/
reward-memory maps on a 40x40 map. The agent has no global notion of where it
has been or where threats are heading. An LSTM hidden state lets the policy
build an *implicit* internal map and enemy-tracking state over time, so it can
prefer unexplored directions, avoid revisiting regions, and remember where
Separatist ships have been spotted.

This script:
    - imports `OuterRimEnv` from `environment.py` (clean, no side-effect code)
    - wraps it in a `DummyVecEnv` (required by SB3 / sb3-contrib)
    - constructs a `RecurrentPPO` model with the `MultiInputLstmPolicy`
      (necessary because the observation is a `Dict` space)
    - trains for `TOTAL_TIMESTEPS` steps with periodic checkpointing
    - saves the final model under `Training_Star_Wars_Galaxy_Phase_2/`

Observation space (Phase 2, vision_radius=7):
    "vision"        : Box(0, 4, (15, 15), uint8)   — terrain + Republic ships
    "danger"        : Box(0.0, 1.0, (15, 15), f32)  — Separatist threat heatmap
    "reward_memory" : Box(0.0, inf, (15, 15), f32)  — planet reward halo map

Comparison with toy_sim_sw_2 (Phase 1):
    Phase 1 obs: {"vision": (5,5) uint8, "seen_memory": (5,5) uint8}
    Phase 2 obs: {"vision": (15,15) uint8, "danger": (15,15) f32,
                  "reward_memory": (15,15) f32}
"""

import os

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

from environment import OuterRimEnv


# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR       = os.path.join(PROJECT_ROOT, "Training_Star_Wars_Galaxy_Phase_2")
LOG_DIR            = os.path.join(TRAINING_DIR, "logs_recurrent")
CHECKPOINT_DIR     = os.path.join(TRAINING_DIR, "Saved RL Models", "RecurrentPPO_checkpoints")
FINAL_MODEL_PATH   = os.path.join(TRAINING_DIR, "Saved RL Models", "RecurrentPPO_Model_Star_Wars_Galaxy")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Total environment steps the agent will see during training.
# Reduced from 1_000_000 to 200_000 for practical training time in Phase 2.
# (At ~50 SPS in the dynamic env, 1M steps ≈ 5+ hours; 200k ≈ ~67 min.)
TOTAL_TIMESTEPS    = 200_000

# Save a checkpoint every this many steps so we can recover / inspect
# intermediate behaviour.
CHECKPOINT_FREQ    = 50_000

# Number of Separatist and Republic ships in the environment.
N_SEP = 2
N_REP = 2


def make_env():
    """Factory that returns a fresh `OuterRimEnv` instance.

    DummyVecEnv requires a callable that builds the env so it can construct
    the vectorised wrapper.
    """
    return OuterRimEnv(N_SEP, N_REP)


def main():
    # -----------------------------------------------------------------------
    # 1) Build the vectorised environment.
    #    DummyVecEnv with a single sub-env keeps things simple while still
    #    satisfying SB3's batch-of-envs contract.
    # -----------------------------------------------------------------------
    env = DummyVecEnv([make_env])

    # -----------------------------------------------------------------------
    # 2) Build the Recurrent PPO model.
    #
    #    Policy choice:
    #      - "MultiInputLstmPolicy" is the recurrent equivalent of
    #        "MultiInputPolicy". It is required here because OuterRimEnv's
    #        observation space is a `Dict`
    #        ({"vision": ..., "danger": ..., "reward_memory": ...}).
    #
    #    Hyperparameters chosen to mirror the feed-forward PPO setup in train.py
    #    so the comparison between PPO and RecurrentPPO is fair:
    #      - n_steps = 2048       : rollout length per env per update
    #      - batch_size = 512     : minibatch size for the SGD updates
    #      - n_epochs = 10        : passes over each rollout
    #      - learning_rate = 3e-4 : Adam LR (SB3 default for PPO)
    #      - clip_range = 0.2     : standard PPO clip
    #      - ent_coef = 0.02      : entropy bonus (encourages exploration)
    #      - gae_lambda = 0.95    : GAE smoothing factor
    #      - vf_coef = 0.5        : value loss coefficient
    #      - gamma = 0.98         : discount factor
    #
    #    Recurrent-specific knobs left at defaults
    #    (single LSTM layer, 256 hidden units, shared between actor and critic).
    # -----------------------------------------------------------------------
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.02,
        gae_lambda=0.95,
        vf_coef=0.5,
        gamma=0.98,
    )

    # -----------------------------------------------------------------------
    # 3) Set up periodic checkpointing so partial progress isn't lost.
    # -----------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="recurrent_ppo",
    )

    # -----------------------------------------------------------------------
    # 4) Train.
    # -----------------------------------------------------------------------
    print(f"[train_recurrent_ppo] Starting training for {TOTAL_TIMESTEPS:,} timesteps.")
    print(f"[train_recurrent_ppo] TensorBoard logs   -> {LOG_DIR}")
    print(f"[train_recurrent_ppo] Checkpoints        -> {CHECKPOINT_DIR}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="RecurrentPPO",
    )

    # -----------------------------------------------------------------------
    # 5) Save the final model.
    # -----------------------------------------------------------------------
    model.save(FINAL_MODEL_PATH)
    print(f"[train_recurrent_ppo] Done. Final model saved to: {FINAL_MODEL_PATH}.zip")

    env.close()


if __name__ == "__main__":
    main()
