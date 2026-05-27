"""
Frontier-Based Exploration for OuterRimEnv Phase 2 (dynamic entities).

Classic robotics exploration loop:
    1. Maintain an explored/unexplored map (via env.seen_map).
    2. Compute frontiers: explored cells adjacent to unexplored cells.
    3. Prefer visiting already-spotted unvisited planets (high-value targets);
       otherwise drive toward the nearest frontier.
    4. Pathfind with A* (Manhattan heuristic) on the 4-connected grid.
    5. Replan every step so newly-revealed information is used immediately.

Deterministic algorithm - no training.

Key differences from Phase 1 (toy_sim_sw_2):
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - Vision radius is 7 (vs 2), so initial exploration covers more area
    - info dict includes "damages" key
    - Separatist ships can interrupt navigation; algorithm is purely
      reactive (does not explicitly model threats) — this is intentional,
      serving as a baseline that ignores danger signals
"""

import heapq
import numpy as np
import pygame

from environment import OuterRimEnv


# Number of ships in the environment
N_SEP = 2
N_REP = 2

# Action layout in OuterRimEnv.step:
ACTION_UP = 0      # row -= 1  ("Forward")
ACTION_DOWN = 1    # row += 1  ("Backward")
ACTION_LEFT = 2    # col -= 1
ACTION_RIGHT = 3   # col += 1

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_to_action(current, neighbor):
    cr, cc = current
    nr, nc = neighbor
    if nr < cr:
        return ACTION_UP
    if nr > cr:
        return ACTION_DOWN
    if nc < cc:
        return ACTION_LEFT
    if nc > cc:
        return ACTION_RIGHT
    return None


def a_star(start, goal, num_rows, num_cols):
    """A* on a 4-connected grid where every in-bounds cell is passable."""
    if start == goal:
        return [start]

    open_heap = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
    came_from = {start: None}
    g_score = {start: 0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        if g > g_score[current]:
            continue

        for dr, dc in NEIGHBORS_4:
            nb = (current[0] + dr, current[1] + dc)
            if not (0 <= nb[0] < num_rows and 0 <= nb[1] < num_cols):
                continue
            tentative_g = g + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (f, tentative_g, nb))

    return None


def find_frontiers(seen_map):
    """Return all frontier cells: seen cells adjacent to at least one unseen cell."""
    num_rows, num_cols = seen_map.shape
    frontiers = []
    seen_rows, seen_cols = np.where(seen_map)
    for r, c in zip(seen_rows, seen_cols):
        for dr, dc in NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < num_rows and 0 <= nc < num_cols and not seen_map[nr, nc]:
                frontiers.append((int(r), int(c)))
                break
    return frontiers


class FrontierExplorer:
    """Greedy nearest-frontier explorer that biases toward known planets."""

    def __init__(self, env):
        self.env = env
        self.target = None
        self.path = []

    def _known_unvisited_planets(self):
        """Cells we've seen that still contain an un-collected planet ('#')."""
        seen = self.env.seen_map
        planet_mask = (self.env.map == "#") & seen
        rows, cols = np.where(planet_mask)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]

    def select_target(self, agent_pos):
        # Priority 1 - nearest known unvisited planet (high reward).
        planets = self._known_unvisited_planets()
        if planets:
            return min(planets, key=lambda p: manhattan(agent_pos, p))

        # Priority 2 - nearest frontier (boundary of the explored region).
        frontiers = find_frontiers(self.env.seen_map)
        if not frontiers:
            return None
        return min(frontiers, key=lambda f: manhattan(agent_pos, f))

    def _target_still_useful(self, agent_pos):
        if self.target is None or self.target == agent_pos:
            return False
        tr, tc = self.target
        cell = self.env.map[tr, tc]
        # Planet target is good as long as it's not yet collected.
        if cell == "#":
            return True
        # Frontier target is good while it still borders unseen cells.
        if not self.env.seen_map[tr, tc]:
            return True
        num_rows, num_cols = self.env.seen_map.shape
        for dr, dc in NEIGHBORS_4:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < num_rows and 0 <= nc < num_cols and not self.env.seen_map[nr, nc]:
                return True
        return False

    def get_action(self):
        agent_pos = self.env.state

        if not self._target_still_useful(agent_pos) or not self.path or self.path[0] != agent_pos:
            self.target = self.select_target(agent_pos)
            if self.target is None:
                return None
            self.path = a_star(agent_pos, self.target,
                               self.env.num_rows, self.env.num_cols)
            if not self.path or len(self.path) < 2:
                return None

        next_cell = self.path[1]
        action = step_to_action(agent_pos, next_cell)
        self.path = self.path[1:]
        return action


def run_episode(env, render=True, max_steps=10_000, frame_delay_ms=0):
    obs, _ = env.reset()
    explorer = FrontierExplorer(env)
    done = False
    episode_score = 0.0
    steps = 0
    info = {"visited_planets": 0, "total_planets": env.total_planets, "damages": 0}

    while not done and steps < max_steps:
        if render:
            env.render()
            if frame_delay_ms > 0:
                pygame.time.delay(frame_delay_ms)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

        action = explorer.get_action()
        if action is None:
            # Fully explored, time still ticking - sit and let mission end.
            action = ACTION_UP

        obs, reward, done, truncated, info = env.step(action)
        episode_score += reward
        steps += 1

    total_cells = env.num_rows * env.num_cols
    explored = int(np.sum(env.seen_map))
    coverage = explored / total_cells * 100.0
    if info["total_planets"]:
        detection_rate = info["visited_planets"] / info["total_planets"] * 100.0
    else:
        detection_rate = 0.0

    return {
        "score": episode_score,
        "steps": steps,
        "coverage": coverage,
        "explored_cells": explored,
        "total_cells": total_cells,
        "planets_found": info["visited_planets"],
        "total_planets": info["total_planets"],
        "detection_rate": detection_rate,
        "damages": info.get("damages", 0),
    }


def main():
    env = OuterRimEnv(N_SEP, N_REP)
    episodes = 5
    results = []

    for ep in range(1, episodes + 1):
        print(f"\n=== Episode {ep}/{episodes} ===")
        r = run_episode(env, render=True)
        if r is None:
            print("Window closed - stopping early.")
            break
        results.append(r)
        print(f"  Score          : {r['score']:.1f}")
        print(f"  Steps          : {r['steps']}")
        print(f"  Coverage       : {r['coverage']:.1f}% "
              f"({r['explored_cells']}/{r['total_cells']} cells)")
        print(f"  Planets Found  : {r['planets_found']}/{r['total_planets']}")
        print(f"  Detection Rate : {r['detection_rate']:.1f}%")
        print(f"  Damages Taken  : {r['damages']}")

    env.close()

    if results:
        avg_coverage = float(np.mean([r["coverage"] for r in results]))
        avg_detection = float(np.mean([r["detection_rate"] for r in results]))
        avg_planets = float(np.mean([r["planets_found"] for r in results]))
        avg_score = float(np.mean([r["score"] for r in results]))
        avg_steps = float(np.mean([r["steps"] for r in results]))
        avg_damages = float(np.mean([r["damages"] for r in results]))

        print("\n========== Frontier Exploration Summary (Phase 2) ==========")
        print(f"  Episodes Run         : {len(results)}")
        print(f"  Avg Coverage         : {avg_coverage:.1f}%")
        print(f"  Avg Planets Found    : {avg_planets:.2f}/{results[0]['total_planets']}")
        print(f"  Avg Detection Rate   : {avg_detection:.1f}%")
        print(f"  Avg Score            : {avg_score:.1f}")
        print(f"  Avg Steps / Episode  : {avg_steps:.0f}")
        print(f"  Avg Damages Taken    : {avg_damages:.2f}")
        print("=============================================================")


if __name__ == "__main__":
    main()
