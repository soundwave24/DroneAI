"""
test_ppo.py  (toy_sim_sw_3_eny — dynamic entity environment)
=============================================================

Evaluate the original PPO agent (trained by train.py) on the Phase 2
OuterRimEnv with mobile Separatist and Republic ships.

Usage:
    python test_ppo.py
    python test_ppo.py --no-render --episodes 10
    python test_ppo.py --n-sep 3 --n-rep 3
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import OuterRimEnv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Training_Star_Wars_Galaxy_Phase_2",
    "Saved RL Models",
    "PPO_Model_Star_Wars_Galaxy_1M",
)

# Number of ships (must match the training configuration)
N_SEP = 2
N_REP = 2


def evaluate(model_path: str = DEFAULT_MODEL_PATH, episodes: int = 5,
             render: bool = True, n_sep: int = N_SEP, n_rep: int = N_REP):

    # Build a live (non-vectorised) env for step-by-step evaluation
    env = OuterRimEnv(n_sep, n_rep)

    # Load model — the saved model was trained through a VecNormalize wrapper,
    # so we need to reconstruct a compatible vec env for PPO.load()
    vec_env = DummyVecEnv([lambda: OuterRimEnv(n_sep, n_rep)])
    model = PPO.load(model_path, env=vec_env)
    print(f"[test_ppo] Loaded model from: {model_path}.zip")

    coverages, detection_rates, planets_found = [], [], []
    rewards, lengths, damages_list = [], [], []

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
        total_pl = int(info.get("total_planets", 0))
        visited = int(info.get("visited_planets", 0))
        detection_rate = 100.0 * visited / total_pl if total_pl > 0 else 0.0
        dmg = int(info.get("damages", 0))

        coverages.append(coverage)
        detection_rates.append(detection_rate)
        planets_found.append(visited)
        rewards.append(ep_reward)
        lengths.append(ep_len)
        damages_list.append(dmg)

        print(
            f"Episode {ep:2d} | return={ep_reward:8.2f} | "
            f"planets={visited:2d}/{total_pl:2d} | "
            f"detection={detection_rate:5.1f}% | "
            f"coverage={coverage:5.1f}% | "
            f"damages={dmg} | steps={ep_len}"
        )

    print("\n--- Aggregate over {} episodes ---".format(episodes))
    print(f"Mean return         : {np.mean(rewards):8.2f}  (std {np.std(rewards):.2f})")
    print(f"Mean planets found  : {np.mean(planets_found):8.2f}  (std {np.std(planets_found):.2f})")
    print(f"Mean detection rate : {np.mean(detection_rates):8.2f}%  (std {np.std(detection_rates):.2f})")
    print(f"Mean coverage       : {np.mean(coverages):8.2f}%  (std {np.std(coverages):.2f})")
    print(f"Mean damages        : {np.mean(damages_list):8.2f}  (std {np.std(damages_list):.2f})")

    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--n-sep", type=int, default=N_SEP,
                   help="Number of Separatist ships.")
    p.add_argument("--n-rep", type=int, default=N_REP,
                   help="Number of Republic ships.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        n_sep=args.n_sep,
        n_rep=args.n_rep,
    )
