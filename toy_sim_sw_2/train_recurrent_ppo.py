"""
train_recurrent_ppo.py
=======================

Train a Recurrent PPO (PPO + LSTM) agent on the `OuterRimEnv` environment.

Why a recurrent policy?
-----------------------
`OuterRimEnv` gives the agent only a small local 5x5 vision patch plus the
local `seen_memory` patch. The agent has no global notion of where it has been
on the 40x40 map. An LSTM hidden state lets the policy build an *implicit*
internal map of explored areas over time, so it can prefer unexplored
directions and avoid revisiting regions that fall outside its vision window.

This script:
    - imports `OuterRimEnv` from `main.py`
    - wraps it in a `DummyVecEnv` (required by SB3 / sb3-contrib)
    - constructs a `RecurrentPPO` model with the `MultiInputLstmPolicy`
      (necessary because the observation is a `Dict` space)
    - trains for `TOTAL_TIMESTEPS` steps with periodic checkpointing
    - saves the final model under `Training_Star_Wars_Galaxy_Phase_1/`
"""

import os

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

# Import the environment from main.py as required by the task spec.
# NOTE: main.py currently executes additional code at import time
# (instantiates an env, runs an episode, loads a saved model). If that
# becomes a problem during training, wrap main.py's runtime code in an
# `if __name__ == "__main__":` block.
from main import OuterRimEnv


# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR       = os.path.join(PROJECT_ROOT, "Training_Star_Wars_Galaxy_Phase_1")
LOG_DIR            = os.path.join(TRAINING_DIR, "logs_recurrent")
CHECKPOINT_DIR     = os.path.join(TRAINING_DIR, "Saved RL Models", "RecurrentPPO_checkpoints")
FINAL_MODEL_PATH   = os.path.join(TRAINING_DIR, "Saved RL Models", "RecurrentPPO_Model_Star_Wars_Galaxy")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Total environment steps the agent will see during training.
TOTAL_TIMESTEPS    = 1_000_000

# Save a checkpoint every this many steps so we can recover / inspect
# intermediate behaviour.
CHECKPOINT_FREQ    = 50_000


def make_env():
    """Factory that returns a fresh `OuterRimEnv` instance.

    DummyVecEnv requires a callable that builds the env so it can construct
    the vectorised wrapper.
    """
    return OuterRimEnv()


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
    #        observation space is a `Dict` ({"vision": ..., "seen_memory": ...}).
    #
    #    Hyperparameters chosen to mirror the feed-forward PPO setup in main.py
    #    so the comparison between PPO and RecurrentPPO is fair:
    #      - n_steps = 4096       : rollout length per env per update
    #      - batch_size = 1024    : minibatch size for the SGD updates
    #      - n_epochs = 15        : passes over each rollout
    #      - learning_rate = 3e-4 : Adam LR (SB3 default for PPO)
    #      - clip_range = 0.2     : standard PPO clip
    #      - ent_coef = 0.01      : entropy bonus (encourages exploration)
    #      - gae_lambda = 0.95    : GAE smoothing factor
    #      - vf_coef = 0.5        : value loss coefficient
    #
    #    Recurrent-specific knobs left at defaults
    #    (single LSTM layer, 256 hidden units, shared between actor and critic).
    # -----------------------------------------------------------------------
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=4096,
        batch_size=1024,
        n_epochs=15,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        gae_lambda=0.95,
        vf_coef=0.5,
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
