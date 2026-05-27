"""
Evaluation script for the Rainbow DQN agent trained on OuterRimEnv.

Loads the saved Rainbow checkpoint, runs N episodes with pygame rendering,
and prints performance metrics (coverage, detection rate, enemies found,
return, episode length).
"""

import os
# By default, render with a real pygame window. To run headless, set
#   SDL_VIDEODRIVER=dummy  in the calling environment.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import time

import numpy as np
import torch

from main import OuterRimEnv
from train_rainbow_dqn import RainbowAgent, pack_obs


def evaluate(model_path: str, episodes: int = 5, render: bool = True,
             render_delay_ms: int = 30, deterministic: bool = True):
    env = OuterRimEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    obs_shape = (2, 5, 5)
    agent = RainbowAgent(obs_shape=obs_shape, n_actions=int(env.action_space.n),
                         device=device)
    agent.load(model_path, map_location=device)
    agent.online.eval()
    # In eval mode NoisyLinear uses the mean weights only -> deterministic
    # unless we explicitly want stochastic acting (deterministic=False).
    if not deterministic:
        agent.online.train()

    total_enemies = 0
    total_total_enemies = 0
    total_coverage = 0.0
    total_return = 0.0
    total_length = 0

    print("=" * 70)
    print(f"Evaluating Rainbow DQN over {episodes} episode(s)")
    print("=" * 70)

    for ep in range(1, episodes + 1):
        obs_dict, _ = env.reset()
        obs = pack_obs(obs_dict)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            if render:
                env.render()
                if render_delay_ms > 0:
                    import pygame
                    pygame.time.delay(render_delay_ms)

            action = agent.act(obs)
            next_obs_dict, reward, done, truncated, info = env.step(action)
            obs = pack_obs(next_obs_dict)
            ep_return += reward
            ep_len += 1
            if truncated:
                break

        explored = int(np.sum(env.seen_map))
        total_cells = env.num_rows * env.num_cols
        coverage = 100.0 * explored / total_cells
        visited = info.get("visited_enemies", 0)
        n_enemies = info.get("total_enemies", 0)
        detection = (100.0 * visited / n_enemies) if n_enemies > 0 else 0.0

        total_enemies += visited
        total_total_enemies += n_enemies
        total_coverage += coverage
        total_return += ep_return
        total_length += ep_len

        print(f"[ep {ep:2d}] return={ep_return:8.2f}  "
              f"enemies={visited:>2d}/{n_enemies:<2d}  "
              f"detection={detection:5.1f}%  "
              f"coverage={coverage:5.1f}%  "
              f"len={ep_len}")

    env.close()

    print("-" * 70)
    avg_return = total_return / episodes
    avg_enemies = total_enemies / episodes
    avg_coverage = total_coverage / episodes
    avg_length = total_length / episodes
    overall_detection = (100.0 * total_enemies / total_total_enemies) if total_total_enemies > 0 else 0.0
    print(f"Average return:        {avg_return:.2f}")
    print(f"Average enemies found: {avg_enemies:.2f} / 20")
    print(f"Average detection rate:{overall_detection:5.1f}%")
    print(f"Average coverage:      {avg_coverage:5.1f}%")
    print(f"Average episode len:   {avg_length:.1f}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str,
                   default=os.path.join("Training_Star_Wars_Galaxy_Phase_1",
                                        "Saved RL Models",
                                        "RainbowDQN_Star_Wars_Galaxy.pt"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true",
                   help="Disable pygame rendering (useful for headless metrics-only runs).")
    p.add_argument("--render-delay-ms", type=int, default=30,
                   help="Delay between frames (slow rendering for visibility).")
    p.add_argument("--stochastic", action="store_true",
                   help="Keep noisy nets in training-mode noise during action selection.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Trained model not found at {args.model_path}. "
            f"Run train_rainbow_dqn.py first."
        )
    evaluate(model_path=args.model_path,
             episodes=args.episodes,
             render=not args.no_render,
             render_delay_ms=args.render_delay_ms,
             deterministic=not args.stochastic)
