# toy_sim_sw_2 - History

## 2026-05-23 - Rename Planet → Enemy
- Renamed all "Planet"/"planet"/"planets" references to "Enemy"/"enemy"/"enemies" across all Python files and history.md in `toy_sim_sw_2/` and `toy_sim_sw_3_eny/`.
- Variable renames: `num_planets→num_enemies`, `planet_positions→enemy_positions`, `visited_planets→visited_enemies`, `total_planets→total_enemies`, `discovered_planet→discovered_enemy`, `planet_reward_map→enemy_reward_map`, `planet_reward_scale→enemy_reward_scale`, `planet_reward_decay→enemy_reward_decay`, `_inject_planet_reward→_inject_enemy_reward`, `planet_mask→enemy_mask`, etc.
- Comments, docstrings, and string literals updated consistently.
- Code logic and functionality fully preserved.

## 2026-05-23 - SAC-Discrete (Task #3)
- Added `train_sac.py` and `test_sac.py` in `toy_sim_sw_2/`.
- Algorithm: **SAC-Discrete** (Christodoulou, 2019, arXiv:1910.07207) — adapts SAC to discrete action spaces. SB3's built-in `SAC` only supports `Box` actions, so the agent is implemented from scratch in PyTorch.
  - `CategoricalActor`: MLP -> logits -> log_softmax over the 4 directional actions. Returns `(probs, log_probs)` so all expectations are computed analytically (no reparameterization needed for discrete).
  - `TwinQ`: two MLP critics, each outputting Q-values for every action (`shape=[batch, 4]`). Target net is a copy with Polyak averaging (`tau=0.005`).
  - Critic target: `r + gamma * (1-done) * sum_a pi(a|s') * (min(Q1_t, Q2_t)(s',a) - alpha * log pi(a|s'))`.
  - Actor loss: `E_s [ sum_a pi(a|s) * (alpha * log pi(a|s) - min(Q1, Q2)(s,a)) ]`.
  - Auto-tuned temperature `alpha` with `target_entropy = 0.98 * log|A|` (Christodoulou's discrete-SAC convention). `log_alpha` is the optimization variable.
  - Replay buffer: numpy circular buffer (`buffer_size=200k`, `learning_starts=5k`, `batch_size=256`).
- Observation handling: `obs_to_vec` flattens the `Dict({vision, seen_memory})` into a single 50-dim float vector (`vision/3` + `seen_memory`). Hidden size 256, ReLU, two hidden layers.
- Training (`train_sac.py`):
  - `total_timesteps=150_000`, `train_freq=1`, `gradient_steps=1`, `gamma=0.99`. CUDA used when available.
  - Logs every 1000 env steps: SPS, 20-episode trailing return / enemies / coverage, `alpha`, policy entropy.
  - Checkpoints every 25k steps to `Training_Star_Wars_Galaxy_Phase_1/Saved RL Models/SAC_Discrete_Model_Star_Wars_Galaxy_<step>_steps.pt`; final model to `..._Galaxy.pt`.
  - Sets `SDL_VIDEODRIVER=dummy` so no pygame window opens during training (rendering is unnecessary while training).
- Evaluation (`test_sac.py`):
  - Loads checkpoint, runs N episodes (default 5) with `env.render()`, drains pygame events each frame so the window stays responsive.
  - Per-episode metrics: return, enemies found / total, detection rate %, coverage % (cells / 1600), length. Prints mean ± std aggregate.
  - CLI flags: `--model`, `--episodes`, `--no-render`, `--stochastic`, `--delay` (per-frame ms).
- Import: `from main import OuterRimEnv` — relies on `main.py`'s existing `if __name__ == "__main__":` guard so import is side-effect-free.
- Smoke-tested: 2k-step training run on CUDA completes cleanly (SPS≈40 during learning), saves a checkpoint, and `test_sac.py --no-render` loads + runs episodes without errors. Full convergence requires the default 150k-step run.

## 2026-05-23 - PPO + Intrinsic Curiosity Module (Task #1)
- Added `train_ppo_curiosity.py` and `test_ppo_curiosity.py` in `toy_sim_sw_2/`.
- ICM (Pathak et al., 2017):
  - `ICMNetwork` (PyTorch `nn.Module`): MLP encoder phi(s) on flattened, normalised `vision`+`seen_memory`; inverse head (phi(s),phi(s'))->a logits; forward head (phi(s),one_hot(a))->phi(s'). `FEATURE_DIM=128`, `hidden=256`.
  - Intrinsic reward = 0.5 * ||forward(phi(s),a) - phi(s')||^2 .
  - ICM loss = (1-beta)*CE(inverse) + beta*MSE(forward), `beta=0.2`. phi(s') is detached in forward loss to avoid trivial encoder collapse.
- Integration with stable-baselines3 PPO:
  - `ICMRewardWrapper(VecEnvWrapper)`: at `step_wait` computes intrinsic reward (no grad), adds `0.05 * intrinsic` onto extrinsic reward, exposes per-step intrinsic/extrinsic values via `info`, and buffers `(obs, next_obs, action)` for ICM training.
  - `ICMTrainingCallback(BaseCallback)`: every `train_freq=4096` env steps (one PPO rollout), runs `ICM_EPOCHS=4` SGD passes with `batch_size=256` on the buffered transitions, then clears the buffer. Logs `icm/inverse_loss`, `icm/forward_loss`, `icm/buffer_size` to TensorBoard.
  - Policy: `MultiInputPolicy` (required for `Dict` obs space), same hyperparameters as `train.py` / `train_recurrent_ppo.py` for fair comparison: `n_steps=4096`, `batch_size=1024`, `n_epochs=15`, `lr=3e-4`, `clip_range=0.2`, `ent_coef=0.01`, `gae_lambda=0.95`, `vf_coef=0.5`.
  - Total timesteps: 1,000,000; `CheckpointCallback` saves every 50k steps to `Training_Star_Wars_Galaxy_Phase_1/Saved RL Models/PPO_Curiosity_checkpoints/`.
- Final outputs: `PPO_Curiosity_Model_Star_Wars_Galaxy.zip` (PPO weights) and `PPO_Curiosity_Model_Star_Wars_Galaxy_icm.pt` (ICM weights). TensorBoard log dir: `Training_Star_Wars_Galaxy_Phase_1/logs_curiosity/`.
- `test_ppo_curiosity.py`:
  - Loads the trained PPO model and runs 5 rendered evaluation episodes with `deterministic=True`.
  - Per-episode metrics: coverage (%), detection rate (%), enemies found, episode score, steps. Prints mean ± std over the run.
- Import: follows `train_recurrent_ppo.py` and does `from main import OuterRimEnv` directly. `main.py` still runs demo code at import time; adding an `if __name__ == "__main__":` guard around the bottom block would speed up startup, but is out of scope for this task.
- Syntax-checked with `python3 -m py_compile`; not yet trained.

## 2026-05-23 - Frontier-Based Exploration baseline (test_frontier.py)
- Added `test_frontier.py`: deterministic classic-robotics exploration baseline for `OuterRimEnv`.
- Algorithm:
  - Maintains explored region via `env.seen_map`.
  - `find_frontiers` returns seen cells with >=1 unseen 4-neighbour.
  - `FrontierExplorer.select_target` prefers nearest known-but-uncollected enemy (`#`); falls back to nearest frontier (Manhattan).
  - `a_star` (Manhattan heuristic, 4-connected) computes the path; replan every step so new vision is used.
  - Action mapping matches `OuterRimEnv.step`: 0=Up, 1=Down, 2=Left, 3=Right.
- Runs 5 evaluation episodes with `env.render()`; prints per-episode and aggregate metrics: coverage %, enemies found, detection rate, score, steps.
- No training required.
- Import: relies on `main.py`'s `if __name__ == "__main__":` guard so `from main import OuterRimEnv` is side-effect-free.

## 2026-05-23 - Coverage Path Planning baseline (test_coverage_path.py) — Task #6
- Added `test_coverage_path.py`: deterministic systematic-coverage baseline for `OuterRimEnv`.
- Algorithm: **Boustrophedon (lawnmower) sweep** + **greedy nearest-enemy collection**.
  - Sweep rows spaced `2*vision_radius + 1 = 5` apart (rows 37, 32, 27, 22, 17, 12, 7, 2) so the 5×5 vision tiles the grid with no gaps.
  - Sweep cols range cols 2..37; with vision radius 2 this covers cols 0..39.
  - Phase 1: traverse boustrophedon waypoints (~336 steps) — guarantees 100% area coverage.
  - Phase 2: greedy nearest-Manhattan navigation to all enemies observed during sweep.
  - Phase 3: idle-against-top-wall until 1000-step mission timer expires.
  - Tracks observed enemies through env's 5×5 vision (does not peek at full map outside the agent's sensor model).
- Import: extracts `OuterRimEnv` from `main.py` via AST (filters to imports + class definition) since `main.py` currently has no `__main__` guard and runs demo + PPO eval at module load.
- Results over 5 evaluation episodes (rendered, 40×40 grid, 20 enemies):
  - Coverage: 100.0% (every cell observed)
  - Detection rate: 100.0% (20/20 enemies visited)
  - Average score: 1064.78 / 1000 steps
- No training required.

## 2026-05-23 - Rainbow DQN (train_rainbow_dqn.py / test_rainbow_dqn.py) — Task #4
- Added `train_rainbow_dqn.py` and `test_rainbow_dqn.py` implementing **full Rainbow DQN** (Hessel et al., 2018) for `OuterRimEnv`.
- Components (all 6 Rainbow ingredients):
  1. **Double DQN** — online net picks the next-state argmax, target net evaluates it.
  2. **Dueling architecture** — separate V(s) and A(s,a) streams reduced to Q via `V + A - mean(A)`.
  3. **Prioritized Experience Replay** — proportional sampling backed by a `SumTree`; priorities updated from per-sample cross-entropy loss; importance-sampling weights with β annealed 0.4 → 1.0.
  4. **N-step returns** — n=3 multi-step Bellman target; deque-based aggregation with episode-end flush.
  5. **Noisy Networks** — factorised Gaussian `NoisyLinear` replaces ε-greedy; noise reset every `act()` and `learn()` call. Eval uses mean weights for determinism (`--stochastic` flag overrides).
  6. **Distributional RL (C51)** — 51 atoms over `[-10, 30]`, projected Bellman target with categorical cross-entropy loss.
- Network: 2-channel Conv (vision + seen_memory) → 2 conv layers (32, 64) → dueling NoisyLinear heads → C51 distribution.
- Observation packing: `vision` (0..3) normalised to [0,1], `seen_memory` (0/1) stacked as channel-2 → (2, 5, 5) float32.
- Hyperparameters: lr=1e-4, γ=0.99, n=3, batch=64, buffer=100k, target update every 1000 steps, train_freq=4 (one gradient step per 4 env steps), PER α=0.5, β-frames = total_timesteps.
- Refactored `main.py`: wrapped the bottom-of-file demo / PPO eval block under `if __name__ == "__main__":` so `from main import OuterRimEnv` is side-effect-free for future scripts too.
- Headless training via `SDL_VIDEODRIVER=dummy` env var (set automatically in the training script).
- Default training: 80k timesteps, model saved to `Training_Star_Wars_Galaxy_Phase_1/Saved RL Models/RainbowDQN_Star_Wars_Galaxy.pt`.
- Evaluation: `test_rainbow_dqn.py` runs N rendered episodes and prints per-episode + aggregate coverage / detection rate / enemies-found / return / length.
- Training run (80k steps, ~30 min on RTX-class CUDA, ~45 fps end-to-end):
  - Best mid-training window (step 53k–62k): ~2.2 enemies / episode, return ≈ -57.
  - Slight late-training drift down to ~1.2 enemies by step 80k (likely buffer/over-fit on a narrow exploration mode; could be mitigated with more diverse exploration or longer training).
- **Evaluation (10 episodes, deterministic / noisy-net mean weights):**
  - Average enemies found: **2.00 / 20** (detection rate 10.0%)
  - Average coverage: **24.6%**
  - Average return: -105.30, episode length 1000 (mission-time cap)
  - Best episode: 5 / 20 enemies, return -25.9.
  - For reference: coverage-path baseline reaches 100% / 100%; Rainbow learns substantially better than random (~1 enemy) but is far from systematic baselines.
