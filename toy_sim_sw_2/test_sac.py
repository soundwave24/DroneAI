"""
Evaluation script for the SAC-Discrete agent trained on OuterRimEnv.

Loads the model produced by train_sac.py, runs a configurable number of
episodes with rendering, and prints per-episode + aggregate metrics
(coverage, detection rate, enemies found, return, length).

Run:
    python test_sac.py
"""

import argparse
import os
import time

import numpy as np
import pygame
import torch

from main import OuterRimEnv
from train_sac import SACDiscrete, SACConfig, obs_to_vec, obs_dim_for, DEVICE


DEFAULT_MODEL_PATH = os.path.join(
    "Training_Star_Wars_Galaxy_Phase_1",
    "Saved RL Models",
    "SAC_Discrete_Model_Star_Wars_Galaxy.pt",
)


def evaluate(
    model_path: str = DEFAULT_MODEL_PATH,
    episodes: int = 5,
    render: bool = True,
    deterministic: bool = True,
    render_delay_ms: int = 20,
):
    env = OuterRimEnv()
    n_actions = env.action_space.n
    obs_dim = obs_dim_for(env)

    cfg = SACConfig()  # only used for network sizing; weights overwrite the rest
    agent = SACDiscrete(obs_dim, n_actions, cfg)
    agent.load(model_path)
    agent.actor.eval()
    print(f"[SAC-Discrete eval] loaded {model_path}  device={DEVICE}")

    returns = []
    enemies_found = []
    coverages = []
    detection_rates = []
    episode_lengths = []

    for ep in range(1, episodes + 1):
        obs_dict, _ = env.reset()
        obs_vec = obs_to_vec(obs_dict)
        done = trunc = False
        ep_return = 0.0
        ep_len = 0

        while not (done or trunc):
            if render:
                # Drain pygame events so the window stays responsive.
                for _ in pygame.event.get():
                    pass
                env.render()
                if render_delay_ms > 0:
                    pygame.time.delay(render_delay_ms)

            action = agent.select_action(obs_vec, deterministic=deterministic)
            obs_dict, reward, done, trunc, info = env.step(action)
            obs_vec = obs_to_vec(obs_dict)
            ep_return += reward
            ep_len += 1

        explored = int(np.sum(env.seen_map))
        total_cells = env.num_rows * env.num_cols
        coverage = 100.0 * explored / total_cells
        total_enemies = info.get("total_enemies", 0)
        visited = info.get("visited_enemies", 0)
        detection_rate = 100.0 * visited / total_enemies if total_enemies > 0 else 0.0

        returns.append(ep_return)
        enemies_found.append(visited)
        coverages.append(coverage)
        detection_rates.append(detection_rate)
        episode_lengths.append(ep_len)

        print(
            f"Episode {ep:2d} | return={ep_return:8.2f} | "
            f"enemies={visited:2d}/{total_enemies:2d} | "
            f"detection={detection_rate:5.1f}% | "
            f"coverage={coverage:5.1f}% ({explored}/{total_cells}) | "
            f"length={ep_len}"
        )

    print("\n--- Aggregate over {} episodes ---".format(episodes))
    print(f"Mean return        : {np.mean(returns):8.2f}  (std {np.std(returns):.2f})")
    print(f"Mean enemies found : {np.mean(enemies_found):8.2f}  (std {np.std(enemies_found):.2f})")
    print(f"Mean detection rate: {np.mean(detection_rates):8.2f}%")
    print(f"Mean coverage      : {np.mean(coverages):8.2f}%")
    print(f"Mean episode length: {np.mean(episode_lengths):8.2f}")

    if render:
        # Hold the final frame briefly so the user can inspect it.
        time.sleep(0.5)
        pygame.quit()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                   help="Path to the saved SAC-Discrete checkpoint (.pt).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true",
                   help="Disable rendering (useful for headless evaluation).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy instead of using argmax.")
    p.add_argument("--delay", type=int, default=20,
                   help="Per-frame delay in ms during rendering.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        render_delay_ms=args.delay,
    )
