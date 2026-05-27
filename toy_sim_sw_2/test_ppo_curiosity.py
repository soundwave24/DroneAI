"""
test_ppo_curiosity.py
=====================

Load the PPO + ICM model trained by `train_ppo_curiosity.py` and run a few
evaluation episodes with rendering.

Reported per-episode metrics:
    - coverage       : fraction of the 40x40 grid the agent has seen
    - detection rate : fraction of enemies located
    - enemies found  : raw enemy count
    - episode score  : sum of extrinsic rewards across the episode

Note: ICM is irrelevant at evaluation time, because we only need the trained
policy to select actions and the intrinsic reward played its part during
training.
"""

import os
import numpy as np

from stable_baselines3 import PPO

# See note in `train_ppo_curiosity.py`: importing `main` executes runtime code.
from main import OuterRimEnv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "Training_Star_Wars_Galaxy_Phase_1")
MODEL_PATH   = os.path.join(TRAINING_DIR, "Saved RL Models", "PPO_Curiosity_Model_Star_Wars_Galaxy")

N_EPISODES = 5


def evaluate_one_episode(env: OuterRimEnv, model: PPO, render: bool = True):
    obs, _ = env.reset()
    done = False
    score = 0.0
    steps = 0

    while not done:
        if render:
            env.render()
        # `deterministic=True` so we measure the policy's preferred behaviour
        # rather than its stochastic exploration noise.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        steps += 1

    total_cells = env.num_rows * env.num_cols
    explored_cells = int(np.sum(env.seen_map))
    coverage = explored_cells / total_cells

    visited = info["visited_enemies"]
    total = info["total_enemies"]
    detection = visited / total if total > 0 else 0.0

    return {
        "score":     score,
        "steps":     steps,
        "coverage":  coverage,
        "detection": detection,
        "enemies":   visited,
        "total":     total,
    }


def main() -> None:
    env = OuterRimEnv()
    print(f"[test_ppo_curiosity] loading model from: {MODEL_PATH}.zip")
    model = PPO.load(MODEL_PATH, env=env)

    coverage_log, detection_log, enemy_log, score_log = [], [], [], []

    for ep in range(1, N_EPISODES + 1):
        result = evaluate_one_episode(env, model, render=True)

        coverage_log.append(result["coverage"])
        detection_log.append(result["detection"])
        enemy_log.append(result["enemies"])
        score_log.append(result["score"])

        print(
            f"Episode {ep:>2}: "
            f"score={result['score']:8.2f}  "
            f"steps={result['steps']:>4}  "
            f"enemies={result['enemies']}/{result['total']}  "
            f"coverage={result['coverage']*100:5.1f}%  "
            f"detection={result['detection']*100:5.1f}%"
        )

    print("\n--- Summary over", N_EPISODES, "episodes ---")
    print(f"Mean episode score  : {np.mean(score_log):.2f} (+/- {np.std(score_log):.2f})")
    print(f"Mean coverage       : {np.mean(coverage_log)*100:.2f}% (+/- {np.std(coverage_log)*100:.2f}%)")
    print(f"Mean detection rate : {np.mean(detection_log)*100:.2f}% (+/- {np.std(detection_log)*100:.2f}%)")
    print(f"Mean enemies found  : {np.mean(enemy_log):.2f} (+/- {np.std(enemy_log):.2f})")

    env.close()


if __name__ == "__main__":
    main()
