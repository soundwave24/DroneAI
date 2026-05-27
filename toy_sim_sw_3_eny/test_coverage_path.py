"""
Coverage Path Planning for OuterRimEnv Phase 2 (dynamic entities).

Implements a deterministic boustrophedon (lawnmower) sweep that exploits the
agent's vision radius (7 → 15x15 vision) to cover the full 40x40 grid with
minimal overlap. No training required.

With vision_radius=7, each row position sees a strip of height/width 15.
The sweep spacing is therefore 15 cells, giving ~3 horizontal sweeps to cover
the full 40-row grid (vs ~9 sweeps for vision_radius=2 in Phase 1).

Key differences from Phase 1 (toy_sim_sw_2):
    - Env constructor: OuterRimEnv(n_sep, n_rep)
    - vision_radius=7 → larger sweep stride, fewer passes needed
    - info dict includes "damages" key
    - Separatist ships may intercept the agent mid-sweep; the algorithm is a
      pure coverage planner and does not avoid threats (baseline behaviour)
"""

import numpy as np
import pygame

from environment import OuterRimEnv


# Number of ships in the environment
N_SEP = 2
N_REP = 2

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

    For vision_radius=7: spacing=15, so the 40-row grid yields sweeps at
    rows ~37, 22, 7 (bottom-up from start), covering all 40 rows.
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
def observe_planets(env, observed):
    """Scan the current 15x15 vision and update the observed-planet set.

    Adds any visible unvisited planets ('#') to `observed`, and removes any
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
    info = {"visited_planets": 0, "total_planets": env.total_planets, "damages": 0}
    observed_planets = set()

    # Initial scan from the start position
    observe_planets(env, observed_planets)

    # --- Phase 1: deterministic boustrophedon sweep ----------------------
    for action in sweep_actions:
        if done or truncated:
            break
        if not _render_and_pump(env, render, frame_delay_ms):
            return None
        obs, reward, done, truncated, info = env.step(int(action))
        episode_score += reward
        steps_taken += 1
        observe_planets(env, observed_planets)

    # --- Phase 2: greedy nearest-planet collection -----------------------
    # The sweep guarantees every cell has been observed at least once, so
    # `observed_planets` now contains every unvisited planet still on the map.
    while not done and not truncated and observed_planets:
        cur = env.state
        target = min(
            observed_planets,
            key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]),
        )
        while env.state != target and not done and not truncated:
            if not _render_and_pump(env, render, frame_delay_ms):
                return None
            action = step_action_toward(env.state, target)
            obs, reward, done, truncated, info = env.step(action)
            episode_score += reward
            steps_taken += 1
            observe_planets(env, observed_planets)
        observed_planets.discard(target)

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
        info["visited_planets"] / info["total_planets"]
        if info["total_planets"] > 0 else 0.0
    )

    return {
        "score": episode_score,
        "steps": steps_taken,
        "coverage": coverage,
        "detection_rate": detection_rate,
        "planets_found": info["visited_planets"],
        "total_planets": info["total_planets"],
        "damages": info.get("damages", 0),
    }


def main():
    num_episodes = 5
    render = True
    frame_delay_ms = 10  # 0 to run as fast as possible

    env = OuterRimEnv(N_SEP, N_REP)

    print("=" * 64)
    print("Coverage Path Planning — Boustrophedon Sweep (Phase 2)")
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
            f"planets={result['planets_found']}/{result['total_planets']}  "
            f"steps={result['steps']}  "
            f"score={result['score']:.2f}  "
            f"damages={result['damages']}"
        )

    if results:
        avg_cov = np.mean([r["coverage"] for r in results]) * 100
        avg_det = np.mean([r["detection_rate"] for r in results]) * 100
        avg_pl = np.mean([r["planets_found"] for r in results])
        avg_score = np.mean([r["score"] for r in results])
        avg_damages = np.mean([r["damages"] for r in results])
        print("-" * 64)
        print(
            f"AVERAGE over {len(results)} episodes: "
            f"coverage={avg_cov:.1f}%  "
            f"detection={avg_det:.1f}%  "
            f"planets={avg_pl:.2f}/{results[0]['total_planets']}  "
            f"score={avg_score:.2f}  "
            f"damages={avg_damages:.2f}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()
