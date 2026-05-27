"""
test_recurrent_ppo.py  (toy_sim_sw_3_eny — Phase 2, dynamic entities)
=======================================================================

Evaluate a trained Recurrent PPO (PPO + LSTM) agent on OuterRimEnv Phase 2.

This script:
    - imports OuterRimEnv from environment.py (clean import, no side effects)
    - loads the trained RecurrentPPO model produced by train_recurrent_ppo.py
    - runs NUM_EPISODES evaluation episodes with on-screen rendering
    - prints per-episode and averaged performance metrics:
          * coverage         (% of the 40x40 grid the agent has seen)
          * detection rate   (% of planets actually visited)
          * planets found    (raw count out of total)
          * episode reward
          * damages taken    (from SeperatistShip collisions)

LSTM bookkeeping during inference
----------------------------------
With a recurrent policy we must thread the LSTM hidden state across timesteps
and tell the policy when a new episode has started so the hidden state is reset.
That is the job of the `lstm_states` and `episode_starts` variables below.

Key differences from toy_sim_sw_2:
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - Info keys: visited_planets, total_planets, damages
    - Obs keys: vision, danger, reward_memory (15x15 each)
"""

import argparse
import os
import numpy as np

from sb3_contrib import RecurrentPPO

from environment import OuterRimEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SEP = 2
N_REP = 2

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR  = os.path.join(PROJECT_ROOT, "Training_Star_Wars_Galaxy_Phase_2")
MODEL_PATH    = os.path.join(TRAINING_DIR, "Saved RL Models",
                              "RecurrentPPO_Model_Star_Wars_Galaxy")

NUM_EPISODES  = 5
DETERMINISTIC = True   # argmax policy (no sampling noise) during evaluation


def evaluate(model_path: str = MODEL_PATH, render: bool = True):
    # -----------------------------------------------------------------------
    # 1) Build environment.
    # -----------------------------------------------------------------------
    env = OuterRimEnv(N_SEP, N_REP)

    # -----------------------------------------------------------------------
    # 2) Load the trained Recurrent PPO model.
    # -----------------------------------------------------------------------
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            "Run train_recurrent_ppo.py first."
        )
    print(f"[test_recurrent_ppo] Loading model: {model_path}.zip")
    model = RecurrentPPO.load(model_path, env=env)

    # -----------------------------------------------------------------------
    # 3) Evaluation loop.
    # -----------------------------------------------------------------------
    coverages       = []
    detection_rates = []
    planets_found   = []
    rewards         = []
    damages_list    = []

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        # LSTM state: None → policy uses zero-initialised hidden state
        lstm_states = None
        # True for the very first step so the policy resets its hidden state.
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            if render:
                env.render()

            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=DETERMINISTIC,
            )
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            episode_starts = np.zeros((1,), dtype=bool)

        # ---- Compute metrics ------------------------------------------------
        total_cells    = env.num_rows * env.num_cols
        explored_cells = int(np.sum(env.seen_map))
        coverage       = (explored_cells / total_cells) * 100.0
        total_pl       = int(info["total_planets"])
        visited_pl     = int(info["visited_planets"])
        detection_rate = (visited_pl / total_pl * 100.0) if total_pl > 0 else 0.0
        damages        = info.get("damages", 0)

        coverages.append(coverage)
        detection_rates.append(detection_rate)
        planets_found.append(visited_pl)
        rewards.append(episode_reward)
        damages_list.append(damages)

        print(
            f"[Episode {episode:>2d}] "
            f"reward={episode_reward:8.2f} | "
            f"steps={step_count:4d} | "
            f"coverage={coverage:5.1f}% ({explored_cells}/{total_cells}) | "
            f"detection={detection_rate:5.1f}% | "
            f"planets={visited_pl}/{total_pl} | "
            f"damages={damages}"
        )

    # -----------------------------------------------------------------------
    # 4) Aggregate results.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"Averaged over {NUM_EPISODES} episodes (RecurrentPPO — Phase 2):")
    print(f"  Mean reward          : {np.mean(rewards):.2f}  (std {np.std(rewards):.2f})")
    print(f"  Mean coverage        : {np.mean(coverages):.2f}%  (std {np.std(coverages):.2f}%)")
    print(f"  Mean detection rate  : {np.mean(detection_rates):.2f}%  (std {np.std(detection_rates):.2f}%)")
    print(f"  Mean planets found   : {np.mean(planets_found):.2f}")
    print(f"  Mean damages taken   : {np.mean(damages_list):.2f}")
    print("=" * 68)

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate RecurrentPPO on OuterRimEnv Phase 2")
    p.add_argument("--model", type=str, default=MODEL_PATH,
                   help="Path to checkpoint (without .zip).")
    p.add_argument("--no-render", action="store_true",
                   help="Disable pygame rendering.")
    p.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = p.parse_args()
    NUM_EPISODES = args.episodes
    evaluate(model_path=args.model, render=not args.no_render)
