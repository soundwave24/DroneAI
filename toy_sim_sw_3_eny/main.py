# Import Gymnasium-related dependencies
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Import Stable Baselines3-related dependencies
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import pygame-related dependencies
import pygame

# Import helper dependencies
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Import local classes
from entity_manager import EntityManager
from npc_ships import SeperatistShip, RepublicShip

class OuterRimEnv(Env):
    def __init__(self, n_sep, n_rep):
        # --- related to world's state spaces --------------------------------------------------------
        self.map = self.generate_map()
        self.num_rows, self.num_cols = self.map.shape
        self.start_position = tuple(np.argwhere(self.map == 'S')[0])
        self.state = self.start_position        # Initialising the initial state of the RL Environment
        self.n_sep = n_sep                      # number of separatist ships
        self.n_rep = n_rep                      # number of republic ships

        # --- related to RL agent, observation spaces and action spaces ----------------------------------------------------
        self.action_space = Discrete(4)

        # Giving the RL agent, 'vision' around it
        # Vision radius: e.g. 1 → 3x3 grid (center + 1 square in each direction)
        self.vision_radius = 7
        obs_height = obs_width = 2 * self.vision_radius + 1

        self.living_penalty = 0.03  # tiny cost per step to discourage camping

        # Each cell is one of: empty space (' '), enemy ('#'), visited enemy ('.'), start ('S'), RepublicShip ('R')
        # Map characters as integers → you can define a vocab for it
        self.char_to_int = {' ': 0, '#': 1, '.': 2, 'S': 3, 'R': 4}

        # Observation: local grid (3x3) with integer values, plus mission time
        self.observation_space = Dict({
            "vision"       : Box(low=0,   high=4,         shape=(obs_height, obs_width), dtype=np.uint8),
            "danger"       : Box(low=0.0, high=1.0,       shape=(obs_height, obs_width), dtype=np.float32),
            "reward_memory": Box(low=0.0, high=np.inf,    shape=(obs_height, obs_width), dtype=np.float32)
        })


        #  --- mission state -----------------------------------------------
        self.mission_time_before_self_destruct = 1000

        # To be used in reward function to give the RL agent a reward for seeing an unexplored pixel
        # for the first time. Creating a copy of the map to mark regions that are seen or not.
        self.seen_map = np.zeros((self.num_rows, self.num_cols), dtype=bool)
        self.agent_damage_taken = False
        self.damages = 0  # counts number of times agent has been damaged


        # --- entities -----------------------------------------------------
        self.entities = []               # list of ALL MobileEntity
        self.agent = self                # optional alias if you treat env as agent wrapper
        self.entity_mgr = EntityManager(self, n_rep=self.n_rep, n_sep=self.n_sep)     # define number of 'SeparatistShips' and 'RepublicShips' being created


        # --- persistent maps -------------------------------------------------
        self.discovered_enemy = np.zeros((self.num_rows, self.num_cols), dtype=bool)
        self.enemy_reward_map = np.zeros((self.num_rows, self.num_cols), dtype=np.float32)

        # tunables
        self.enemy_reward_scale = 5.0     # reward at the enemy tile itself
        self.enemy_reward_decay = 1.5     # larger → slower fall-off


        # --- pygame -----------------------------------------------
        pygame.init()
        self.cell_size = 20  # reduce cell size to fit 40x40 on screen
        self.screen = pygame.display.set_mode(
            (self.num_cols * self.cell_size, self.num_rows * self.cell_size)
        )
        pygame.display.set_caption("Star Wars Galaxy Explorer (Phase 2)")


    ####################
    # Helper functions #
    ####################
    def generate_map(self, rows=40, cols=40, num_enemies=20):
        map = np.full((rows, cols), " ", dtype='<U1')

        # Randomly choose N enemy positions (excluding start) and keeping the number of enemies in each
        # episode constant
        available_positions = [(i, j) for i in range(rows) for j in range(cols) if (i, j) != (39, 21)]
        enemy_positions = random.sample(available_positions, num_enemies)

        for i, j in enemy_positions:
            map[i, j] = '#'

        map[39, 21] = 'S'
        return map

    def get_RL_agent_local_observation(self):
        r, c = self.state
        v = self.vision_radius
        size = 2*v + 1

        obs = np.zeros((size, size), dtype=np.uint8)
        rm  = np.zeros_like(obs, dtype=np.float32)
        dang = np.zeros_like(obs, dtype=np.float32)

        # --- a. vision -------------------------------
        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                    cell_char = self.map[rr, cc]
                    obs[dr + v, dc + v] = self.char_to_int.get(cell_char, 0)
                else:
                    obs[dr + v, dc + v] = 0

        # --- b. transient danger map -------------------------------------
        for ship in self.entities:
            if not isinstance(ship, SeperatistShip):
                continue
            sr, sc = ship.pos
            # relative position of the ship w.r.t agent
            rel_r, rel_c = sr - r + v, sc - c + v
            if not (0 <= rel_r < size and 0 <= rel_c < size):
                continue           # ship is outside vision → no penalty

            for dr in range(-v, v + 1):
                for dc in range(-v, v + 1):
                    dist = abs(dr) + abs(dc)          # Manhattan distance
                    if dist > v:                      # outside view
                        continue
                    # heavier penalty near the ship, linearly decaying
                    penalty = (v - dist + 1) / (v + 1)    # 1.0 at dis = 0 → ≈ 0.09 at edge
                    cell_r, cell_c = rel_r + dr, rel_c + dc
                    if 0 <= cell_r < size and 0 <= cell_c < size:
                        penalty = (v - dist + 1) / (v + 1)
                        dang[cell_r, cell_c] = max(dang[cell_r, cell_c], penalty)

        # --- c. reward memory map -------------------------------
        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                    rm[dr+v, dc+v] = self.enemy_reward_map[rr, cc]

        # --- d. show RepublicShips  -------------------------------------
        for ship in self.entities:
            if isinstance(ship, RepublicShip):
                sr, sc = ship.pos              # ship row / col on the big map
                rel_r, rel_c = sr - r + v, sc - c + v   # coordinates inside vision window
                if 0 <= rel_r < size and 0 <= rel_c < size:
                    obs[rel_r, rel_c] = self.char_to_int['R']

        return {
            "vision": obs,          # Current visual snapshot (local terrain)
            "danger": dang,         # Danger map
            "reward_memory": rm     # Agent's remembered "explored" map
        }

    def _inject_enemy_reward(self, pos, sign=+1.0):
        """Add (+1) or remove (−1) this enemy's contribution from enemy_reward_map."""
        pr, pc = pos
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                dist = abs(pr - r) + abs(pc - c)   # Manhattan
                contrib = sign * self.enemy_reward_scale / (1.0 + self.enemy_reward_decay*dist)
                self.enemy_reward_map[r, c] += contrib

    def check_valid_position(self, position):
        row, col = position

        # If RL agent goes out of the map
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        return True

    def is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return (abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1)


    ###############################################################
    # OpenAI Gymnasium and Stable Baselines3's required functions #
    ###############################################################
    def step(self, action):
        # --- Decrease 'mission_time_before_self_destruct' time -------------------------------------
        self.mission_time_before_self_destruct -= 1

        # --- Apply RL agent action -----------------------------------------------------------------
        new_pos = np.array(self.state)
        if action == 0:     # Forward
            new_pos[0] -= 1
        elif action == 1:   # Backward
            new_pos[0] += 1
        elif action == 2:   # Leftward
            new_pos[1] -= 1
        elif action == 3:   # Rightward
            new_pos[1] += 1

        # Check if RL agent is in a valid position
        if self.check_valid_position(new_pos):
            if all(tuple(new_pos) != e.pos for e in self.entities):
                self.state = tuple(new_pos)

        #########################################
        # Calculate Reward with Reward Function #
        #########################################
        reward = 0

        row, col = self.state
        r, c     = self.state
        v        = self.vision_radius

        exploration_reward = 0

        # --- Penalise points for living -------------------------------------
        reward -= self.living_penalty

        # --- Reward points for every enemy visited -------------------------------------
        if self.map[row, col] == '#':       # If a enemy is visited
            self._inject_enemy_reward((row, col), sign=-1.0)  # erase its halo
            reward += 50
            self.map[row, col] = '.'        # Mark enemy as visited, so RL agent dosent choose to stay there infinitely and force it to find other enemies
            self.visited_enemies += 1

        # --- Penalise points if agent steps into a previously seen region or at the starting position --------------------
        if self.seen_map[row, col]:  # already seen
            reward -= 0.3

        # Penalty for stepping back onto the starting position
        if (row, col) == self.start_position:
            reward -= 0.3

        # --- Penalise points for revisitng an already visited enemy -------------------------------------
        if self.map[row, col] == '.':
            reward -= 1.0  # or some stronger penalty

        # --- Reward points for newly explored (first-time seen) cells in vision --------------------------
        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                    if not self.seen_map[rr, cc]:
                        self.seen_map[rr, cc] = True
                        exploration_reward += 0.3  # reward per new cell seen

        # Discover new enemies that just entered vision
        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                    if self.map[rr, cc] == '#' and not self.discovered_enemy[rr, cc]:
                        self.discovered_enemy[rr, cc] = True
                        self._inject_enemy_reward((rr, cc), sign=+1.0)   # add its reward field

        reward += exploration_reward * 1.5
        reward += self.enemy_reward_map[row, col]        # dense, cumulative

        # --- Penalise points for camping near a corner (2x2 area) -------------------------------------
        if (row <= 1 and col <= 1) or \
        (row <= 1 and col >= self.num_cols - 2) or \
        (row >= self.num_rows - 2 and col <= 1) or \
        (row >= self.num_rows - 2 and col >= self.num_cols - 2):
            reward -= 0.1

        #######################################################################################################
        # Handling movement logic and rewards/penalty points related to 'SeparatistShips' and 'RepublicShips' #
        #######################################################################################################
        # --- a. Move NPCs ('SeparatistShips' and 'RepublicShips') -------------------------------------
        proposals = {}   # pos -> entity list
        for e in self.entities:
            move = e.choose_action_stochastic(self)
            new_pos = e.propose_move(move)
            # keep inside bounds
            r,c = new_pos
            if not (0 <= r < self.num_rows and 0 <= c < self.num_cols):
                new_pos = e.pos               # bounce
            proposals.setdefault(new_pos, []).append(e)

        # --- b. Resolve collisions (single occupant rule + adjacency kill) -------------------------------------
        #   * single occupant rule
        #   * tie‑breaker: Separatist ⇒ victory & occupies; Republic damaged
        #   * Republic–Republic collision → first moves, others stay
        survivors = []
        for dest, group in proposals.items():

            if dest == self.state:
                survivors.extend(group)        # everyone stays where they are
                continue

            if len(group) == 1 and dest != self.state:  # free cell
                group[0].pos = dest
                survivors.append(group[0])
                continue

            # multiple claimants → resolve
            seps = [g for g in group if isinstance(g, SeperatistShip)]
            reps = [g for g in group if isinstance(g, RepublicShip)]

            if seps:  # at least one Separatist present
                # damage republic ships in that cell
                for rep in reps:
                    continue  # simply omit from survivors list
                # one separatist (priority) takes the cell, the rest stay put
                seps[0].pos = dest
                survivors.extend(seps)  # all separatists survive (only first moved)
            else:    # only republics vying for cell
                chosen = random.choice(group)
                chosen.pos = dest
                survivors.append(chosen)
                survivors.extend([g for g in group if g is not chosen])

        # Updating surviving entities
        self.entities = survivors

        # 'SeparatistShip' damage 'RepublicShip' or RL agent by neighbouring
        to_remove = []
        for sep in (e for e in self.entities if isinstance(e, SeperatistShip)):
            for rep in (e for e in self.entities if isinstance(e, RepublicShip)):
                if self.is_adjacent(sep.pos, rep.pos):
                    to_remove.append(rep)

            # Check if RL agent is adjacent to Separatist
            if self.is_adjacent(sep.pos, self.state):
                self.agent_damage_taken = True
                self.damages += 1

        for rep in to_remove:
            if rep in self.entities:
                self.entities.remove(rep)

        # --- c. Penalise points for directly adjacent/neighbouring a 'SeparatistShip' (agent takes damage) ----------------------
        if self.agent_damage_taken:
            reward -= 30
            self.agent_damage_taken = False
            # done = True


        ####################################################################################
        # Handling danger map logic, where when a 'SeparatistShip' enters a agent's vision #
        ####################################################################################
        # --- Penalise points for 'seeing' a 'SeparatistShip' in their vision to encourage/get them to stay away ----------------------
        # a. Build the transient danger map exactly as in get_RL_agent_local_observation
        #    (you may move that code into a helper so you don't duplicate it).
        dang = np.zeros((2*v+1, 2*v+1), dtype=np.float32)
        for ship in self.entities:
            if not isinstance(ship, SeperatistShip):
                continue
            sr, sc = ship.pos
            rel_r, rel_c = sr - r + v, sc - c + v
            if 0 <= rel_r < 2*v+1 and 0 <= rel_c < 2*v+1:
                for dr in range(-v, v + 1):
                    for dc in range(-v, v + 1):
                        dist = abs(dr) + abs(dc)
                        if dist > v:
                            continue

                        cell_r = rel_r + dr
                        cell_c = rel_c + dc

                        if 0 <= cell_r < 2*v+1 and 0 <= cell_c < 2*v+1:
                            penalty = (v - dist + 1) / (v + 1)
                            dang[cell_r, cell_c] = max(dang[cell_r, cell_c], penalty)

        # b. Convert that map into a negative reward every step.
        #    Two common choices:   (pick ONE)
        #    • sum of penalties in the window   → harsher when surrounded
        #    • max penalty (nearest threat)     → distance-only
        danger_strength = np.max(dang)        # np.sum(dang) or np.max(dang)
        danger_scale    = -0.3                # tune sign & magnitude

        reward += danger_scale * danger_strength


        if self.mission_time_before_self_destruct <= 0:
            done = True
        else:
            done = False

        truncated = False
        info = {
            "visited_enemies": self.visited_enemies,
            "total_enemies": self.total_enemies,
            "damages": self.damages
        }

        return self.get_RL_agent_local_observation(), reward, done, truncated, info


    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))

        agent_r, agent_c = self.state
        v = self.vision_radius

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                # If seen_map is True and it's just empty space (i.e. not a enemy or visited)
                if self.seen_map[row, col] and self.map[row, col] == ' ':
                    pygame.draw.rect(self.screen, (255, 200, 200), (cell_left, cell_top, self.cell_size, self.cell_size))

                # Draw the vision radius in yellow (as a background highlight)
                if abs(row - agent_r) <= v and abs(col - agent_c) <= v:
                    pygame.draw.rect(self.screen, (255, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if self.map[row, col] == '#':  # Draw non-visited enemy in Light Blue
                    pygame.draw.rect(self.screen, (173, 216, 230), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.map[row, col] == '.':  # Draw visited enemy in Green
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.map[row, col] == 'S':  # Draw starting position in Black
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if (row, col) == self.state:  # Draw RL agent position in Gray
                    pygame.draw.rect(self.screen, (125, 125, 125), (cell_left, cell_top, self.cell_size, self.cell_size))

        # Draw 'SeparatistShips' and 'RepublicShips'
        for ship in self.entities:
            sr, sc = ship.pos
            color = (255,0,0) if isinstance(ship, SeperatistShip) else (0,0,255)
            pygame.draw.rect(self.screen, color,
                            (sc*self.cell_size, sr*self.cell_size,
                            self.cell_size, self.cell_size))

        # === Highlight vision radius of each RepublicShip (NEUTRAL ZONE) ===
        for ship in self.entities:
            if isinstance(ship, RepublicShip):
                sr, sc = ship.pos
                vr = ship.vision_radius
                for dr in range(-vr, vr + 1):
                    for dc in range(-vr, vr + 1):
                        rr, cc = sr + dr, sc + dc
                        if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                            left = cc * self.cell_size
                            top  = rr * self.cell_size
                            pygame.draw.rect(self.screen, (0, 0, 255), (left, top, self.cell_size, self.cell_size), width=1)

        # === Highlight vision radius of each SeperatistShip (DANGER ZONE) ===
        for ship in self.entities:
            if isinstance(ship, SeperatistShip):
                sr, sc = ship.pos
                vr = ship.vision_radius
                for dr in range(-vr, vr + 1):
                    for dc in range(-vr, vr + 1):
                        rr, cc = sr + dr, sc + dc
                        if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                            left = cc * self.cell_size
                            top  = rr * self.cell_size
                            pygame.draw.rect(self.screen, (255, 0, 0), (left, top, self.cell_size, self.cell_size), width=1)

        pygame.display.update()  # Update the display
        # pygame.time.delay(50)   # Slow down the rendering

    def reset(self, *, seed=None, options=None):
        # --- Generate a new map ------------------------------------------------------------------------
        self.map = self.generate_map(rows=40, cols=40, num_enemies=20)

        # --- Reinitialize dependent properties ---------------------------------------------------------
        self.num_rows, self.num_cols = self.map.shape
        self.seen_map[:]           = False
        self.discovered_enemy[:]  = False
        self.enemy_reward_map[:]  = 0.0

        self.start_position = tuple(np.argwhere(self.map == 'S')[0])
        self.state = self.start_position
        self.mission_time_before_self_destruct = 1000
        self.agent_damage_taken = False
        self.damages = 0

        self.total_enemies = np.sum(self.map == '#')
        self.visited_enemies = 0

        self.entities = []
        self.entity_mgr = EntityManager(self, n_rep=self.n_sep, n_sep=self.n_sep)


        # --- Update Pygame screen if dimensions changed ------------------------------------------------
        self.screen = pygame.display.set_mode(
            (self.num_cols * self.cell_size, self.num_rows * self.cell_size)
        )

        info = {}
        return self.get_RL_agent_local_observation(), info


if __name__ == "__main__":
    # --- Random-action demo ---------------------------------------------------
    env = OuterRimEnv(2, 2)
    episodes = 5
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_score = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_score += reward
        print(f"Episode: {episode} | Score: {episode_score:.2f} | "
              f"Enemies Found: {info['visited_enemies']}/{info['total_enemies']} | "
              f"Damages: {info['damages']}")
    env.close()

    # --- PPO training / evaluation demo ---------------------------------------
    vec_env = DummyVecEnv([lambda: OuterRimEnv(2, 2)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=5.0)

    log_path = os.path.join('Training_Star_Wars_Galaxy_Phase_2', 'logs')
    print(log_path)

    PPO_DRL_model = PPO('MultiInputPolicy',
                        vec_env,
                        verbose=1,
                        tensorboard_log=log_path,
                        normalize_advantage=True,
                        ent_coef=0.02,
                        gamma=0.98,
                        n_steps=2048,
                        gae_lambda=0.95)

    # PPO_DRL_model.learn(total_timesteps=500000)
    # PPO_Model_Custom = os.path.join('Training_Star_Wars_Galaxy_Phase_2',
    #                                 'Saved RL Models', 'PPO_Model_Star_Wars_Galaxy_1M')
    # PPO_DRL_model.save(PPO_Model_Custom)

    PPO_Model_Custom = os.path.join('Training_Star_Wars_Galaxy_Phase_2',
                                    'Saved RL Models', 'PPO_Model_Star_Wars_Galaxy_1M')
    reloaded_PPO_DRL_model = PPO.load(PPO_Model_Custom, env=vec_env)
    print("Loaded PPO DRL model from:", PPO_Model_Custom)

    env2 = OuterRimEnv(5, 5)
    episodes = 5
    total_enemies_found = 0
    for episode in range(1, episodes + 1):
        obs, _ = env2.reset()
        done = False
        episode_score = 0
        while not done:
            env2.render()
            action, _ = reloaded_PPO_DRL_model.predict(obs)
            obs, reward, done, truncated, info = env2.step(action)
            episode_score += reward
        total_enemies_found += info['visited_enemies']
        print(f"Episode: {episode} | Score: {episode_score:.2f} | "
              f"Enemies Found: {info['visited_enemies']}/{info['total_enemies']} | "
              f"Damages: {info['damages']}")
    env2.close()
    average_enemies_found = total_enemies_found / episodes
    print(f"\nAverage enemies found over {episodes} episodes: {average_enemies_found:.2f}")
