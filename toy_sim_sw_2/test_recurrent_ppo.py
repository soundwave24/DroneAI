"""
test_recurrent_ppo.py
======================

Evaluate a trained Recurrent PPO (PPO + LSTM) agent on `OuterRimEnv`.

This script:
    - imports `OuterRimEnv` from `main.py`
    - loads the trained RecurrentPPO model produced by `train_recurrent_ppo.py`
    - runs `NUM_EPISODES` evaluation episodes with on-screen rendering
    - prints per-episode and averaged performance metrics:
          * coverage         (% of the 40x40 grid the agent has seen)
          * detection rate   (% of enemies actually visited)
          * enemies found    (raw count out of total)
          * episode reward

LSTM bookkeeping during inference
---------------------------------
With a non-recurrent policy, `model.predict(obs)` is stateless. With a
recurrent policy we must thread the LSTM hidden state across timesteps and
tell the policy when a new episode has started so the hidden state is reset.
That is the job of the `lstm_states` and `episode_starts` variables below.
"""

import argparse
import os
import numpy as np

from sb3_contrib import RecurrentPPO

from main import OuterRimEnv


# ---------------------------------------------------------------------------
# Paths and evaluation settings
# ---------------------------------------------------------------------------
PROJECT_ROOT      = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR      = os.path.join(PROJECT_ROOT, "Training_Star_Wars_Galaxy_Phase_1")
# Default: use the best available checkpoint (150k steps) since full 1M training
# hasn't been completed yet.  Override with --model to point to a different file.
MODEL_PATH        = os.path.join(TRAINING_DIR, "Saved RL Models",
                                 "RecurrentPPO_checkpoints", "recurrent_ppo_150000_steps")

NUM_EPISODES      = 5
DETERMINISTIC     = True   # use the policy's mean action (no sampling) during eval


def evaluate(model_path: str = MODEL_PATH, render: bool = True):
    # -----------------------------------------------------------------------
    # 1) Build the environment (non-vectorised; we drive it directly).
    # -----------------------------------------------------------------------
    env = OuterRimEnv()

    # -----------------------------------------------------------------------
    # 2) Load the trained Recurrent PPO model.
    # -----------------------------------------------------------------------
    print(f"[test_recurrent_ppo] Loading model from: {model_path}.zip")
    model = RecurrentPPO.load(model_path, env=env)

    # -----------------------------------------------------------------------
    # 3) Run evaluation episodes.
    # -----------------------------------------------------------------------
    coverages       = []
    detection_rates = []
    enemies_found   = []
    rewards         = []

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        # ---- LSTM state bookkeeping --------------------------------------
        # `lstm_states` is the hidden+cell state tuple carried across steps.
        # Start of episode -> None tells the policy to use zero-initialised
        # hidden state.
        lstm_states = None
        # `episode_starts` is a (n_envs,) bool array; True for the first
        # step of a new episode so the policy resets its hidden state.
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            if render:
                env.render()

            # Use deterministic=True for evaluation to remove sampling noise
            # and get a stable picture of what the policy actually learned.
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=DETERMINISTIC,
            )

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # After the very first step, future steps in this episode are
            # NOT episode starts -> hidden state should be carried forward.
            episode_starts = np.zeros((1,), dtype=bool)

        # ---- Compute metrics for this episode ----------------------------
        total_cells     = env.num_rows * env.num_cols
        explored_cells  = int(np.sum(env.seen_map))
        coverage        = (explored_cells / total_cells) * 100.0
        total_enemies   = int(info["total_enemies"])
        visited_enemies = int(info["visited_enemies"])
        detection_rate  = (visited_enemies / total_enemies * 100.0) if total_enemies > 0 else 0.0

        coverages.append(coverage)
        detection_rates.append(detection_rate)
        enemies_found.append(visited_enemies)
        rewards.append(episode_reward)

        print(
            f"[Episode {episode:>2d}] "
            f"reward={episode_reward:8.2f} | "
            f"steps={step_count:4d} | "
            f"coverage={coverage:5.1f}% ({explored_cells}/{total_cells}) | "
            f"detection={detection_rate:5.1f}% | "
            f"enemies={visited_enemies}/{total_enemies}"
        )

    # -----------------------------------------------------------------------
    # 4) Aggregate results.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Averaged over {NUM_EPISODES} episodes:")
    print(f"  Mean reward          : {np.mean(rewards):.2f}  (std {np.std(rewards):.2f})")
    print(f"  Mean coverage        : {np.mean(coverages):.2f}%  (std {np.std(coverages):.2f})")
    print(f"  Mean detection rate  : {np.mean(detection_rates):.2f}%  (std {np.std(detection_rates):.2f})")
    print(f"  Mean enemies found   : {np.mean(enemies_found):.2f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=MODEL_PATH,
                   help="Path to checkpoint (without .zip).")
    p.add_argument("--no-render", action="store_true",
                   help="Disable pygame rendering.")
    args = p.parse_args()
    evaluate(model_path=args.model, render=not args.no_render)
