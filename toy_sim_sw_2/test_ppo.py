"""
test_ppo.py
===========

Evaluate the PPO (original) agent trained on OuterRimEnv.

Usage:
    python test_ppo.py
    python test_ppo.py --no-render --episodes 10
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from main import OuterRimEnv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Training_Star_Wars_Galaxy_Phase_1",
    "Saved RL Models",
    "PPO_Model_Star_Wars_Galaxy_1M",
)


def evaluate(model_path: str = DEFAULT_MODEL_PATH, episodes: int = 5, render: bool = True):
    env = OuterRimEnv()
    # Load model; wrap in DummyVecEnv for SB3 compatibility during predict
    vec_env = DummyVecEnv([lambda: OuterRimEnv()])
    model = PPO.load(model_path, env=vec_env)
    print(f"[test_ppo] Loaded model from: {model_path}.zip")

    coverages, detection_rates, enemies_found, rewards, lengths = [], [], [], [], []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            if render:
                env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

        explored = int(np.sum(env.seen_map))
        total_cells = env.num_rows * env.num_cols
        coverage = 100.0 * explored / total_cells
        total_en = int(info.get("total_enemies", 0))
        visited = int(info.get("visited_enemies", 0))
        detection_rate = 100.0 * visited / total_en if total_en > 0 else 0.0

        coverages.append(coverage)
        detection_rates.append(detection_rate)
        enemies_found.append(visited)
        rewards.append(ep_reward)
        lengths.append(ep_len)

        print(
            f"Episode {ep:2d} | return={ep_reward:8.2f} | "
            f"enemies={visited:2d}/{total_en:2d} | "
            f"detection={detection_rate:5.1f}% | "
            f"coverage={coverage:5.1f}% | "
            f"steps={ep_len}"
        )

    print("\n--- Aggregate over {} episodes ---".format(episodes))
    print(f"Mean return        : {np.mean(rewards):8.2f}  (std {np.std(rewards):.2f})")
    print(f"Mean enemies found : {np.mean(enemies_found):8.2f}  (std {np.std(enemies_found):.2f})")
    print(f"Mean detection rate: {np.mean(detection_rates):8.2f}%  (std {np.std(detection_rates):.2f})")
    print(f"Mean coverage      : {np.mean(coverages):8.2f}%  (std {np.std(coverages):.2f})")
    print(f"Mean episode length: {np.mean(lengths):8.2f}")

    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(model_path=args.model, episodes=args.episodes, render=not args.no_render)
