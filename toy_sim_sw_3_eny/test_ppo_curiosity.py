"""
test_ppo_curiosity.py  (toy_sim_sw_3_eny — Phase 2, dynamic entities)
=======================================================================

Load the PPO + ICM model trained by train_ppo_curiosity.py and run evaluation
episodes with rendering.

Note: The ICM (Intrinsic Curiosity Module) is irrelevant at evaluation time.
Only the trained PPO policy is needed to select actions; the intrinsic reward
played its part exclusively during training.

Reported per-episode metrics:
    - coverage       : fraction of the 40x40 grid the agent has seen
    - detection rate : fraction of planets located (visited_planets / total)
    - planets found  : raw planet count
    - episode score  : sum of extrinsic rewards across the episode
    - damages taken  : SeperatistShip collision count

Key differences from toy_sim_sw_2:
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - Info keys: visited_planets, total_planets, damages
    - Obs keys: vision, danger, reward_memory (15x15 each)
"""

import argparse
import os
import numpy as np
from datetime import datetime

import pygame
from PIL import Image
from stable_baselines3 import PPO

from environment import OuterRimEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SEP = 2
N_REP = 2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "Training_Star_Wars_Galaxy_Phase_2")
MODEL_PATH   = os.path.join(TRAINING_DIR, "Saved RL Models",
                             "PPO_Curiosity_Model_Star_Wars_Galaxy")

N_EPISODES = 5


def evaluate_one_episode(env: OuterRimEnv, model: PPO, render: bool = True,
                         capture_frames: bool = False, render_delay_ms: int = 20):
    obs, _ = env.reset()
    done = False
    score = 0.0
    steps = 0
    frames = []

    while not done:
        if render:
            for _ in pygame.event.get():
                pass
            env.render()

            # Capture frame for GIF
            if capture_frames:
                surf = pygame.display.get_surface()
                frame = pygame.surfarray.array3d(surf)
                # Transpose to get correct orientation
                frame = np.transpose(frame, (1, 0, 2))
                frames.append(frame)

            if render_delay_ms > 0:
                pygame.time.delay(render_delay_ms)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        steps += 1

    total_cells    = env.num_rows * env.num_cols
    explored_cells = int(np.sum(env.seen_map))
    coverage       = explored_cells / total_cells

    visited = info["visited_planets"]
    total   = info["total_planets"]
    damages = info.get("damages", 0)
    detection = visited / total if total > 0 else 0.0

    return {
        "score":     score,
        "steps":     steps,
        "coverage":  coverage,
        "detection": detection,
        "planets":   visited,
        "total":     total,
        "damages":   damages,
        "frames":    frames,
    }


def main(model_path: str = MODEL_PATH, n_episodes: int = N_EPISODES,
         render: bool = True, save_gif: bool = False, gif_dir: str = "ppo_curiosity_gifs",
         render_delay_ms: int = 20) -> None:
    env = OuterRimEnv(N_SEP, N_REP)

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            "Run train_ppo_curiosity.py first."
        )
    print(f"[test_ppo_curiosity] Loading model: {model_path}.zip")
    model = PPO.load(model_path, env=env)

    # Create GIF directory if saving
    if save_gif:
        os.makedirs(gif_dir, exist_ok=True)
        print(f"[test_ppo_curiosity] GIFs will be saved to: {gif_dir}/")

    coverage_log, detection_log, planet_log, score_log, damage_log = [], [], [], [], []

    for ep in range(1, n_episodes + 1):
        result = evaluate_one_episode(env, model, render=render,
                                       capture_frames=save_gif,
                                       render_delay_ms=render_delay_ms)

        coverage_log.append(result["coverage"])
        detection_log.append(result["detection"])
        planet_log.append(result["planets"])
        score_log.append(result["score"])
        damage_log.append(result["damages"])

        print(
            f"Episode {ep:>2}: "
            f"score={result['score']:8.2f}  "
            f"steps={result['steps']:>4}  "
            f"planets={result['planets']}/{result['total']}  "
            f"coverage={result['coverage']*100:5.1f}%  "
            f"detection={result['detection']*100:5.1f}%  "
            f"damages={result['damages']}"
        )

        # Save GIF for this episode
        if save_gif and len(result["frames"]) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(
                gif_dir,
                f"ppo_curiosity_episode_{ep}_{timestamp}_score{result['score']:.0f}.gif"
            )

            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in result["frames"]]

            # Save as GIF
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=render_delay_ms,
                loop=0
            )
            print(f"  → Saved GIF: {gif_path} ({len(result['frames'])} frames)")

    print("\n" + "=" * 68)
    print(f"Summary over {n_episodes} episodes (PPO+Curiosity — Phase 2):")
    print(f"  Mean score           : {np.mean(score_log):.2f} (+/- {np.std(score_log):.2f})")
    print(f"  Mean coverage        : {np.mean(coverage_log)*100:.2f}% (+/- {np.std(coverage_log)*100:.2f}%)")
    print(f"  Mean detection rate  : {np.mean(detection_log)*100:.2f}% (+/- {np.std(detection_log)*100:.2f}%)")
    print(f"  Mean planets found   : {np.mean(planet_log):.2f}")
    print(f"  Mean damages taken   : {np.mean(damage_log):.2f}")
    print("=" * 68)

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate PPO+Curiosity on OuterRimEnv Phase 2")
    p.add_argument("--model", type=str, default=MODEL_PATH,
                   help="Path to checkpoint (without .zip).")
    p.add_argument("--no-render", action="store_true",
                   help="Disable rendering.")
    p.add_argument("--episodes", type=int, default=N_EPISODES,
                   help="Number of episodes to run.")
    p.add_argument("--save-gif", action="store_true",
                   help="Save each episode as a GIF animation.")
    p.add_argument("--gif-dir", type=str, default="ppo_curiosity_gifs",
                   help="Directory to save GIF files (default: ppo_curiosity_gifs).")
    p.add_argument("--delay", type=int, default=20,
                   help="Per-frame delay in ms during rendering (default: 20).")
    args = p.parse_args()
    main(model_path=args.model, n_episodes=args.episodes, render=not args.no_render,
         save_gif=args.save_gif, gif_dir=args.gif_dir, render_delay_ms=args.delay)
