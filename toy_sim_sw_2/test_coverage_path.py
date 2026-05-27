"""
Coverage Path Planning for OuterRimEnv (Task #6).

Implements a deterministic boustrophedon (lawnmower) sweep that exploits the
agent's vision radius (2 → 5x5 vision) to cover the full 40x40 grid with
minimal overlap. No training required.

Note: main.py lacks an `if __name__ == "__main__":` guard and runs a demo +
PPO evaluation at module load. We extract just the OuterRimEnv class via AST
to avoid those side effects.
"""

import ast
import os
import sys

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Import OuterRimEnv from main.py without triggering its top-level demo code
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_PATH = os.path.join(_HERE, "main.py")


def _load_outer_rim_env():
    with open(_MAIN_PATH, "r") as f:
        tree = ast.parse(f.read(), filename=_MAIN_PATH)

    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "OuterRimEnv":
            keep.append(node)

    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"__name__": "main_outer_rim", "__file__": _MAIN_PATH}
    exec(compile(module, _MAIN_PATH, "exec"), ns)
    return ns["OuterRimEnv"]


OuterRimEnv = _load_outer_rim_env()


# ---------------------------------------------------------------------------
# Coverage path planner: boustrophedon sweep
# ---------------------------------------------------------------------------
# Actions in OuterRimEnv:
#   0 = Up (row - 1), 1 = Down (row + 1), 2 = Left (col - 1), 3 = Right (col + 1)
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3


def build_sweep_waypoints(start, num_rows, num_cols, vision_radius):
    """
    Build a boustrophedon path of waypoints.

    With vision radius v, each row position sees a strip of width (2v+1).
    Sweep rows are spaced (2v+1) apart, with the first/last sweep row offset
    by v from the boundary so vision just covers the edge rows.

    The agent traverses col=v to col=(num_cols - 1 - v) on each sweep row, so
    its vision covers col 0 to col (num_cols - 1).
    """
    spacing = 2 * vision_radius + 1
    row_lo = vision_radius
    row_hi = num_rows - 1 - vision_radius
    col_lo = vision_radius
    col_hi = num_cols - 1 - vision_radius

    # Sweep rows ordered bottom-up (closer to start position first)
    sweep_rows = list(range(row_hi, row_lo - 1, -spacing))
    if sweep_rows[-1] != row_lo:
        sweep_rows.append(row_lo)

    waypoints = []
    going_right = True
    for sr in sweep_rows:
        if going_right:
            waypoints.append((sr, col_lo))
            waypoints.append((sr, col_hi))
        else:
            waypoints.append((sr, col_hi))
            waypoints.append((sr, col_lo))
        going_right = not going_right

    return waypoints


def waypoints_to_actions(start, waypoints):
    """Convert a list of (row, col) waypoints into a sequence of grid actions.

    Moves axis-by-axis (row first, then column) from the current position.
    """
    actions = []
    cur_r, cur_c = start
    for tr, tc in waypoints:
        while cur_r != tr:
            if cur_r > tr:
                actions.append(ACTION_UP)
                cur_r -= 1
            else:
                actions.append(ACTION_DOWN)
                cur_r += 1
        while cur_c != tc:
            if cur_c > tc:
                actions.append(ACTION_LEFT)
                cur_c -= 1
            else:
                actions.append(ACTION_RIGHT)
                cur_c += 1
    return actions


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def observe_enemies(env, observed):
    """Scan the current 5x5 vision and update the observed-enemy set.

    Adds any visible unvisited enemies ('#') to `observed`, and removes any
    that have since been visited ('.').
    """
    r, c = env.state
    v = env.vision_radius
    for dr in range(-v, v + 1):
        for dc in range(-v, v + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < env.num_rows and 0 <= cc < env.num_cols:
                cell = env.map[rr, cc]
                if cell == "#":
                    observed.add((rr, cc))
                elif cell == ".":
                    observed.discard((rr, cc))


def step_action_toward(cur, target):
    """Greedy one-step action from `cur` toward `target` (row first, then col)."""
    r, c = cur
    tr, tc = target
    if r > tr:
        return ACTION_UP
    if r < tr:
        return ACTION_DOWN
    if c > tc:
        return ACTION_LEFT
    return ACTION_RIGHT


def _render_and_pump(env, render, frame_delay_ms):
    if not render:
        return True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    env.render()
    if frame_delay_ms > 0:
        pygame.time.delay(frame_delay_ms)
    return True


def run_episode(env, render=True, frame_delay_ms=10):
    obs, _ = env.reset()

    waypoints = build_sweep_waypoints(
        env.start_position, env.num_rows, env.num_cols, env.vision_radius
    )
    sweep_actions = waypoints_to_actions(env.start_position, waypoints)

    episode_score = 0.0
    steps_taken = 0
    done = False
    truncated = False
    info = {"visited_enemies": 0, "total_enemies": env.total_enemies}
    observed_enemies = set()

    # Initial scan from the start position
    observe_enemies(env, observed_enemies)

    # --- Phase 1: deterministic boustrophedon sweep ----------------------
    for action in sweep_actions:
        if done or truncated:
            break
        if not _render_and_pump(env, render, frame_delay_ms):
            return None
        obs, reward, done, truncated, info = env.step(int(action))
        episode_score += reward
        steps_taken += 1
        observe_enemies(env, observed_enemies)

    # --- Phase 2: greedy nearest-enemy collection -----------------------
    # The sweep guarantees every cell has been observed at least once, so
    # `observed_enemies` now contains every unvisited enemy still on the map.
    while not done and not truncated and observed_enemies:
        cur = env.state
        target = min(
            observed_enemies,
            key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]),
        )
        while env.state != target and not done and not truncated:
            if not _render_and_pump(env, render, frame_delay_ms):
                return None
            action = step_action_toward(env.state, target)
            obs, reward, done, truncated, info = env.step(action)
            episode_score += reward
            steps_taken += 1
            observe_enemies(env, observed_enemies)
        observed_enemies.discard(target)

    # --- Phase 3: idle until mission timer expires -----------------------
    while not done and not truncated:
        if not _render_and_pump(env, render, frame_delay_ms):
            return None
        r, _ = env.state
        action = ACTION_UP if r > 0 else ACTION_DOWN
        obs, reward, done, truncated, info = env.step(action)
        episode_score += reward
        steps_taken += 1

    coverage = float(np.sum(env.seen_map)) / (env.num_rows * env.num_cols)
    detection_rate = (
        info["visited_enemies"] / info["total_enemies"]
        if info["total_enemies"] > 0 else 0.0
    )

    return {
        "score": episode_score,
        "steps": steps_taken,
        "coverage": coverage,
        "detection_rate": detection_rate,
        "enemies_found": info["visited_enemies"],
        "total_enemies": info["total_enemies"],
    }


def main():
    num_episodes = 5
    render = True
    frame_delay_ms = 10  # 0 to run as fast as possible

    env = OuterRimEnv()

    print("=" * 64)
    print("Coverage Path Planning — Boustrophedon Sweep")
    print(f"Map: {env.num_rows}x{env.num_cols}, vision_radius={env.vision_radius}")
    print(f"Start: {env.start_position}")
    print("=" * 64)

    # Show planned waypoints once for transparency
    obs, _ = env.reset()
    wps = build_sweep_waypoints(
        env.start_position, env.num_rows, env.num_cols, env.vision_radius
    )
    acts = waypoints_to_actions(env.start_position, wps)
    print(f"Planned waypoints ({len(wps)}): {wps}")
    print(f"Planned action sequence length: {len(acts)} (mission budget: 1000)")
    print("-" * 64)

    results = []
    for ep in range(1, num_episodes + 1):
        result = run_episode(env, render=render, frame_delay_ms=frame_delay_ms)
        if result is None:
            print("Window closed by user — stopping early.")
            break
        results.append(result)
        print(
            f"Episode {ep}: "
            f"coverage={result['coverage']*100:5.1f}%  "
            f"detection={result['detection_rate']*100:5.1f}%  "
            f"enemies={result['enemies_found']}/{result['total_enemies']}  "
            f"steps={result['steps']}  "
            f"score={result['score']:.2f}"
        )

    if results:
        avg_cov = np.mean([r["coverage"] for r in results]) * 100
        avg_det = np.mean([r["detection_rate"] for r in results]) * 100
        avg_pl = np.mean([r["enemies_found"] for r in results])
        avg_score = np.mean([r["score"] for r in results])
        print("-" * 64)
        print(
            f"AVERAGE over {len(results)} episodes: "
            f"coverage={avg_cov:.1f}%  "
            f"detection={avg_det:.1f}%  "
            f"enemies={avg_pl:.2f}/{results[0]['total_enemies']}  "
            f"score={avg_score:.2f}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()
