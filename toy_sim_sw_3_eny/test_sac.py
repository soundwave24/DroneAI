"""
test_sac.py  (toy_sim_sw_3_eny — Phase 2, dynamic entities)
=============================================================

Evaluation script for the SAC-Discrete agent trained on OuterRimEnv Phase 2.

Loads the model produced by train_sac.py, runs a configurable number of
episodes with rendering, and prints per-episode + aggregate metrics
(coverage, detection rate, planets found, return, length, damages).

Key differences from toy_sim_sw_2:
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - Info keys: visited_planets, total_planets, damages
    - obs_to_vec handles 3 channels (vision/4 + danger + tanh(rm/10)) → 675 dim
    - Model path: Training_Star_Wars_Galaxy_Phase_2/

Run:
    python test_sac.py
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pygame
import torch
from PIL import Image

from environment import OuterRimEnv
from train_sac import SACDiscrete, SACConfig, obs_to_vec, obs_dim_for, DEVICE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SEP = 2
N_REP = 2

DEFAULT_MODEL_PATH = os.path.join(
    "Training_Star_Wars_Galaxy_Phase_2",
    "Saved RL Models",
    "SAC_Discrete_Model_Star_Wars_Galaxy.pt",
)


def evaluate(
    model_path: str = DEFAULT_MODEL_PATH,
    episodes: int = 5,
    render: bool = True,
    deterministic: bool = True,
    render_delay_ms: int = 20,
    save_gif: bool = False,
    gif_dir: str = "sac_gifs",
):
    env = OuterRimEnv(N_SEP, N_REP)
    n_actions = env.action_space.n
    obs_dim = obs_dim_for(env)

    cfg = SACConfig()   # used only for network sizing; weights come from checkpoint
    agent = SACDiscrete(obs_dim, n_actions, cfg)
    agent.load(model_path)
    agent.actor.eval()
    print(f"[SAC-Discrete eval] loaded {model_path}  device={DEVICE}")

    # Create GIF directory if saving
    if save_gif:
        os.makedirs(gif_dir, exist_ok=True)
        print(f"[SAC-Discrete eval] GIFs will be saved to: {gif_dir}/")

    returns        = []
    planets_found  = []
    coverages      = []
    detection_rates = []
    episode_lengths = []
    damages_list   = []

    for ep in range(1, episodes + 1):
        obs_dict, _ = env.reset()
        obs_vec = obs_to_vec(obs_dict)
        done = trunc = False
        ep_return = 0.0
        ep_len = 0

        # Frame buffer for GIF
        frames = []

        while not (done or trunc):
            if render:
                for _ in pygame.event.get():
                    pass
                env.render()

                # Capture frame for GIF
                if save_gif:
                    # Get the pygame surface and convert to numpy array
                    surf = pygame.display.get_surface()
                    # pygame uses (width, height, 3) but we need (height, width, 3)
                    frame = pygame.surfarray.array3d(surf)
                    # Transpose to get correct orientation (pygame returns (W,H,3), PIL needs (H,W,3))
                    frame = np.transpose(frame, (1, 0, 2))
                    frames.append(frame)

                if render_delay_ms > 0:
                    pygame.time.delay(render_delay_ms)

            action = agent.select_action(obs_vec, deterministic=deterministic)
            obs_dict, reward, done, trunc, info = env.step(action)
            obs_vec = obs_to_vec(obs_dict)
            ep_return += reward
            ep_len += 1

        explored      = int(np.sum(env.seen_map))
        total_cells   = env.num_rows * env.num_cols
        coverage      = 100.0 * explored / total_cells
        total_pl      = info.get("total_planets", 0)
        visited_pl    = info.get("visited_planets", 0)
        detection_rate = 100.0 * visited_pl / total_pl if total_pl > 0 else 0.0
        damages       = info.get("damages", 0)

        returns.append(ep_return)
        planets_found.append(visited_pl)
        coverages.append(coverage)
        detection_rates.append(detection_rate)
        episode_lengths.append(ep_len)
        damages_list.append(damages)

        print(
            f"Episode {ep:2d} | return={ep_return:8.2f} | "
            f"planets={visited_pl:2d}/{total_pl:<2d} | "
            f"detection={detection_rate:5.1f}% | "
            f"coverage={coverage:5.1f}% ({explored}/{total_cells}) | "
            f"length={ep_len} | "
            f"damages={damages}"
        )

        # Save GIF for this episode
        if save_gif and len(frames) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(
                gif_dir,
                f"sac_episode_{ep}_{timestamp}_score{ep_return:.0f}.gif"
            )

            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]

            # Save as GIF (duration in ms per frame)
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=render_delay_ms,  # ms per frame
                loop=0  # 0 means loop forever
            )
            print(f"  → Saved GIF: {gif_path} ({len(frames)} frames)")
            frames.clear()

    print("\n" + "=" * 68)
    print(f"Aggregate over {episodes} episodes (SAC-Discrete — Phase 2):")
    print(f"  Mean return        : {np.mean(returns):8.2f}  (std {np.std(returns):.2f})")
    print(f"  Mean planets found : {np.mean(planets_found):8.2f}  (std {np.std(planets_found):.2f})")
    print(f"  Mean detection rate: {np.mean(detection_rates):8.2f}%")
    print(f"  Mean coverage      : {np.mean(coverages):8.2f}%")
    print(f"  Mean episode length: {np.mean(episode_lengths):8.2f}")
    print(f"  Mean damages taken : {np.mean(damages_list):8.2f}")
    print("=" * 68)

    if render:
        time.sleep(0.5)
        pygame.quit()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SAC-Discrete on OuterRimEnv Phase 2")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                   help="Path to the saved SAC-Discrete checkpoint (.pt).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", action="store_true",
                   help="Disable rendering.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy instead of using argmax.")
    p.add_argument("--delay", type=int, default=20,
                   help="Per-frame delay in ms during rendering.")
    p.add_argument("--save-gif", action="store_true",
                   help="Save each episode as a GIF animation.")
    p.add_argument("--gif-dir", type=str, default="sac_gifs",
                   help="Directory to save GIF files (default: sac_gifs).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        render_delay_ms=args.delay,
        save_gif=args.save_gif,
        gif_dir=args.gif_dir,
    )
