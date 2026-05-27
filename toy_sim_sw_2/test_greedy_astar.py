"""
Greedy + A* Hybrid algorithm for OuterRimEnv.

Strategy:
  1. Greedy target selection: pick the nearest unexplored cell (or, if any
     known-but-unvisited enemy is in sight, prefer that as the next target).
  2. A* pathfinding: compute shortest grid path from the agent's current
     position to that target (Manhattan heuristic; the world is open grid so
     there are no impassable obstacles, but A* generalises cleanly and lets us
     bias toward exploration along the way).
  3. Execute path step-by-step. Re-plan whenever the target becomes "explored"
     (seen_map already True) or the agent has consumed its current plan.

This is fully deterministic - no RL training required.
"""

import heapq
import numpy as np
import pygame

from main import OuterRimEnv


# Action mapping (matches OuterRimEnv.step):
#   0 = up    (row - 1)
#   1 = down  (row + 1)
#   2 = left  (col - 1)
#   3 = right (col + 1)
DELTA_TO_ACTION = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, num_rows, num_cols):
    """Standard A* on a 4-connected open grid. Returns a list of (r, c) cells
    from `start` (exclusive) to `goal` (inclusive), or None if unreachable."""
    if start == goal:
        return []

    open_heap = [(manhattan(start, goal), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while came_from[node] is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        if g > g_score.get(current, float('inf')):
            continue

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < num_rows and 0 <= nc < num_cols):
                continue
            neighbor = (nr, nc)
            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))

    return None


def find_nearest_unexplored(env):
    """BFS-style search: return the closest cell (by Manhattan distance) that
    has not yet been added to `seen_map`. Ties are broken by row-major order."""
    seen = env.seen_map
    if seen.all():
        return None

    start = env.state
    # Cheap closest-unexplored lookup: compute Manhattan distance to every
    # unexplored cell and take the argmin. For a 40x40 grid this is trivial.
    unexplored = np.argwhere(~seen)
    dists = np.abs(unexplored[:, 0] - start[0]) + np.abs(unexplored[:, 1] - start[1])
    idx = int(np.argmin(dists))
    return tuple(unexplored[idx])


def find_visible_unvisited_enemy(env):
    """If a enemy has been spotted (seen_map True at that cell) but not yet
    visited (still '#' rather than '.'), grab the closest one - it is high
    value to collect now since detours stay cheap."""
    enemy_mask = (env.map == '#') & env.seen_map
    if not enemy_mask.any():
        return None

    enemies = np.argwhere(enemy_mask)
    start = env.state
    dists = np.abs(enemies[:, 0] - start[0]) + np.abs(enemies[:, 1] - start[1])
    idx = int(np.argmin(dists))
    return tuple(enemies[idx])


def select_target(env):
    """Greedy selection: prefer a known-but-uncollected enemy, else go for
    the nearest unexplored cell."""
    enemy = find_visible_unvisited_enemy(env)
    if enemy is not None:
        return enemy, "enemy"
    cell = find_nearest_unexplored(env)
    if cell is not None:
        return cell, "explore"
    return None, "done"


def step_to_action(current, next_cell):
    delta = (next_cell[0] - current[0], next_cell[1] - current[1])
    return DELTA_TO_ACTION[delta]


def run_episode(env, render=True, verbose=True):
    obs, _ = env.reset()
    done = False
    episode_score = 0.0
    target = None
    path = []
    info = {"visited_enemies": 0, "total_enemies": env.total_enemies}
    steps = 0

    while not done:
        if render:
            # Pump pygame events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            env.render()

        # Re-plan if needed: no target yet, target already explored/visited,
        # or path consumed.
        need_replan = (
            target is None
            or not path
            or (env.seen_map[target] and env.map[target] != '#')
        )

        if need_replan:
            target, reason = select_target(env)
            if target is None:
                # Whole map explored - just idle until time runs out by
                # taking a no-op-ish action (move into a wall).
                if verbose:
                    print(f"  [step {steps}] map fully explored; idling")
                action = 0  # will likely be a clamped move
            else:
                path = astar(env.state, target, env.num_rows, env.num_cols)
                if not path:
                    # No path (shouldn't happen on an open grid) - bail to a
                    # random valid action just to keep things moving.
                    action = 0
                    path = []
                else:
                    action = step_to_action(env.state, path[0])
                    path = path[1:]
        else:
            action = step_to_action(env.state, path[0])
            path = path[1:]

        obs, reward, done, truncated, info = env.step(action)
        episode_score += reward
        steps += 1

    return episode_score, info, steps


def main(num_episodes=3, render=True):
    env = OuterRimEnv()

    all_coverages = []
    all_detection_rates = []
    all_enemies_found = []
    all_scores = []

    for episode in range(1, num_episodes + 1):
        print(f"\n=== Episode {episode}/{num_episodes} ===")
        score, info, steps = run_episode(env, render=render, verbose=True)

        total_cells = env.num_rows * env.num_cols
        explored = int(np.sum(env.seen_map))
        coverage = explored / total_cells * 100.0
        visited = info["visited_enemies"]
        total = info["total_enemies"]
        detection = (visited / total * 100.0) if total > 0 else 0.0

        all_coverages.append(coverage)
        all_detection_rates.append(detection)
        all_enemies_found.append(visited)
        all_scores.append(score)

        print(f"  Score:          {score:.2f}")
        print(f"  Steps taken:    {steps}")
        print(f"  Coverage:       {coverage:.1f}% ({explored}/{total_cells} cells)")
        print(f"  Enemies found:  {visited}/{total}")
        print(f"  Detection rate: {detection:.1f}%")

    env.close()

    print("\n=== Greedy + A* Summary ===")
    print(f"Episodes:                 {num_episodes}")
    print(f"Avg score:                {np.mean(all_scores):.2f}")
    print(f"Avg coverage:             {np.mean(all_coverages):.1f}%")
    print(f"Avg detection rate:       {np.mean(all_detection_rates):.1f}%")
    print(f"Avg enemies found/episode:{np.mean(all_enemies_found):.2f}")


if __name__ == "__main__":
    main(num_episodes=3, render=True)
