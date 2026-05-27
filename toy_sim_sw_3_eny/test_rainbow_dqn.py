"""
test_rainbow_dqn.py  (toy_sim_sw_3_eny — Phase 2, dynamic entities)
=====================================================================

Evaluation script for the Rainbow DQN agent trained on OuterRimEnv Phase 2.

Loads the saved Rainbow checkpoint, runs N episodes with pygame rendering,
and prints performance metrics (coverage, detection rate, planets found,
return, episode length, damages taken).

Key differences from toy_sim_sw_2:
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - obs_shape: (3, 15, 15)  — 3 channels × 15×15 (vs (2,5,5) in Phase 1)
    - pack_obs: vision/4 + danger + tanh(reward_memory/10)
    - Info keys: visited_planets, total_planets, damages
    - Model path: Training_Star_Wars_Galaxy_Phase_2/
"""

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import time

import numpy as np
import torch

from environment import OuterRimEnv
from train_rainbow_dqn import RainbowAgent, pack_obs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SEP = 2
N_REP = 2

OBS_SHAPE = (3, 15, 15)   # 3 channels × 15×15 vision patch


def evaluate(model_path: str, episodes: int = 5, render: bool = True,
             render_delay_ms: int = 30, deterministic: bool = True):
    env = OuterRimEnv(N_SEP, N_REP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = RainbowAgent(obs_shape=OBS_SHAPE, n_actions=int(env.action_space.n),
                         device=device)
    agent.load(model_path, map_location=device)
    agent.online.eval()
    # In eval mode NoisyLinear uses mean weights → deterministic actions.
    # Set train() if stochastic behaviour is desired.
    if not deterministic:
        agent.online.train()

    total_planets_found = 0
    total_total_planets = 0
    total_coverage      = 0.0
    total_return        = 0.0
    total_length        = 0
    total_damages       = 0

    print("=" * 72)
    print(f"Evaluating Rainbow DQN over {episodes} episode(s) — Phase 2")
    print("=" * 72)

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

        explored      = int(np.sum(env.seen_map))
        total_cells   = env.num_rows * env.num_cols
        coverage      = 100.0 * explored / total_cells
        visited_pl    = info.get("visited_planets", 0)
        n_planets     = info.get("total_planets", 0)
        detection     = (100.0 * visited_pl / n_planets) if n_planets > 0 else 0.0
        damages       = info.get("damages", 0)

        total_planets_found += visited_pl
        total_total_planets += n_planets
        total_coverage      += coverage
        total_return        += ep_return
        total_length        += ep_len
        total_damages       += damages

        print(f"[ep {ep:2d}] return={ep_return:8.2f}  "
              f"planets={visited_pl:>2d}/{n_planets:<2d}  "
              f"detection={detection:5.1f}%  "
              f"coverage={coverage:5.1f}%  "
              f"len={ep_len}  "
              f"damages={damages}")

    env.close()

    print("-" * 72)
    avg_return    = total_return / episodes
    avg_planets   = total_planets_found / episodes
    avg_coverage  = total_coverage / episodes
    avg_length    = total_length / episodes
    avg_damages   = total_damages / episodes
    overall_det   = (100.0 * total_planets_found / total_total_planets) \
                    if total_total_planets > 0 else 0.0

    print(f"Average return:          {avg_return:.2f}")
    print(f"Average planets found:   {avg_planets:.2f}")
    print(f"Average detection rate:  {overall_det:.1f}%")
    print(f"Average coverage:        {avg_coverage:.1f}%")
    print(f"Average episode length:  {avg_length:.1f}")
    print(f"Average damages taken:   {avg_damages:.2f}")
    print("=" * 72)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Rainbow DQN on OuterRimEnv Phase 2")
    p.add_argument("--model-path", type=str,
                   default=os.path.join("Training_Star_Wars_Galaxy_Phase_2",
                                        "Saved RL Models",
                                        "RainbowDQN_Star_Wars_Galaxy.pt"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true",
                   help="Disable pygame rendering.")
    p.add_argument("--render-delay-ms", type=int, default=30,
                   help="Delay between frames (ms).")
    p.add_argument("--stochastic", action="store_true",
                   help="Keep noisy nets in training-mode noise during action selection.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Trained model not found at {args.model_path}.\n"
            "Run train_rainbow_dqn.py first."
        )
    evaluate(model_path=args.model_path,
             episodes=args.episodes,
             render=not args.no_render,
             render_delay_ms=args.render_delay_ms,
             deterministic=not args.stochastic)
