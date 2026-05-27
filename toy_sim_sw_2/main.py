# Import Gymnasium-related dependencies
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import Stable Baselines3-related dependencies
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Import pygame-related dependencies
import pygame

# Import helper dependencies
import numpy as np
import random
import os

class OuterRimEnv(Env):
    def __init__(self):
        # --- related to world's state spaces --------------------------------------------------------
        self.map = self.generate_map()
        self.num_rows, self.num_cols = self.map.shape
        self.start_position = tuple(np.argwhere(self.map == 'S')[0])
        self.state = self.start_position        # Initialising the initial state of the RL Environment
        

        # --- related to RL agent, observation spaces and action spaces ----------------------------------------------------
        self.action_space = Discrete(4)

        # Giving the RL agent, 'vision' around it
        # Vision radius: 1 → 3x3 grid (center + 1 square in each direction)
        self.vision_radius = 2
        obs_height = obs_width = 2 * self.vision_radius + 1

        # Each cell is one of: empty space (' '), enemy ('#'), visited enemy ('.'), start ('S')
        # Map characters as integers → you can define a vocab for it
        self.char_to_int = {' ': 0, '#': 1, '.': 2, 'S': 3}

        # Observation: local grid (3x3) with integer values, plus mission time
        self.observation_space = Dict({
            "vision": Box(low=0, high=3, shape=(obs_height, obs_width), dtype=np.uint8),
            "seen_memory": Box(low=0, high=1, shape=(obs_height, obs_width), dtype=np.uint8)
        })
        

        #  --- mission state -----------------------------------------------
        self.mission_time_before_self_destruct = 1000

        # To be used in reward function to give the RL agent a reward for seeing an unexplored pixel
        # for the first time. Creating a copy of the map to mark regions that are seen or not.
        self.seen_map = np.zeros((self.num_rows, self.num_cols), dtype=bool)


        #  --- pygame -----------------------------------------------
        pygame.init()
        self.cell_size = 20  # reduce cell size to fit 40x40 on screen
        self.info_panel_width = 250  # Width for performance metrics panel
        self.screen = pygame.display.set_mode(
            (self.num_cols * self.cell_size + self.info_panel_width, self.num_rows * self.cell_size)
        )
        pygame.display.set_caption("Star Wars Galaxy Explorer (Phase 1)")


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
        mem = np.zeros_like(obs, dtype=np.uint8)

        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                        cell_char = self.map[rr, cc]
                        obs[dr + v, dc + v] = self.char_to_int.get(cell_char, 0)
                        mem[dr + v, dc + v] = int(self.seen_map[rr, cc])
                    else:
                        obs[dr + v, dc + v] = 0
                        mem[dr + v, dc + v] = 0

        return {
            "vision": obs,          # Current visual snapshot (local terrain)
            "seen_memory": mem,     # Agent's remembered "explored" map
        }

    def check_valid_position(self, position):
        row, col = position

        # If RL agent goes out of the map
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False
        
        return True


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
            self.state = tuple(new_pos)

        #########################################
        # Calculate Reward with Reward Function #
        #########################################
        reward = 0

        r, c = self.state
        v = self.vision_radius

        exploration_reward = 0

        # --- Reward points for every enemy visited -------------------------------------
        if self.map[r, c] == '#':       # If a enemy is visited
            reward += 50
            self.map[r, c] = '.'        # Mark enemy as visited, so RL agent dosent choose to stay there infinitely and force it to find other enemies
            self.visited_enemies += 1

        # --- Penalise points if agent steps into a previously seen region or at the starting position --------------------
        if self.seen_map[r, c]:  # already seen
            reward -= 0.3

        # Penalty for stepping back onto the starting position
        if (r, c) == self.start_position:
            reward -= 0.3

        # --- Penalise points for revisitng an already visited enemy -------------------------------------
        if self.map[r, c] == '.':
            reward -= 1.0  # or some stronger penalty

        # --- Reward points for newly explored (first-time seen) cells in vision --------------------------
        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.num_rows and 0 <= cc < self.num_cols:
                    if not self.seen_map[rr, cc]:
                        self.seen_map[rr, cc] = True
                        exploration_reward += 0.3  # reward per new cell seen
        
        reward += exploration_reward

        # --- Penalise points for camping near a corner (2x2 area) -------------------------------------
        if (r <= 1 and c <= 1) or \
        (r <= 1 and c >= self.num_cols - 2) or \
        (r >= self.num_rows - 2 and c <= 1) or \
        (r >= self.num_rows - 2 and c >= self.num_cols - 2):
            reward -= 0.1


        if self.mission_time_before_self_destruct <= 0:
            done = True
        else:
            done = False

        truncated = False
        info = {
            "visited_enemies": self.visited_enemies,
            "total_enemies": self.total_enemies
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

                if self.map[row, col] == '#':  # Draw non-visited enemy in Blue
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.map[row, col] == '.':  # Draw visited enemy in Green
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.map[row, col] == 'S':  # Draw starting position in Black
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if (row, col) == self.state:  # Draw RL agent position in Gray
                    pygame.draw.rect(self.screen, (125, 125, 125), (cell_left, cell_top, self.cell_size, self.cell_size))

        # Draw legend
        font = pygame.font.Font(None, 18)
        legend_x = 10
        legend_y = 10
        legend_items = [
            ((125, 125, 125), "Agent"),
            ((0, 0, 255), "Enemy"),
            ((0, 255, 0), "Visited Enemy"),
            ((255, 255, 0), "Vision Radius"),
            ((255, 200, 200), "Explored Area"),
            ((0, 0, 0), "Start Position")
        ]

        for i, (color, label) in enumerate(legend_items):
            y_offset = legend_y + i * 20
            pygame.draw.rect(self.screen, color, (legend_x, y_offset, 15, 15))
            text = font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (legend_x + 20, y_offset))

        # Draw performance metrics panel on the right side
        panel_x = self.num_cols * self.cell_size
        panel_bg_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.num_rows * self.cell_size)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_bg_rect)
        pygame.draw.line(self.screen, (0, 0, 0), (panel_x, 0), (panel_x, self.num_rows * self.cell_size), 2)

        # Calculate metrics
        total_cells = self.num_rows * self.num_cols
        explored_cells = np.sum(self.seen_map)
        coverage = (explored_cells / total_cells) * 100
        detection_rate = (self.visited_enemies / self.total_enemies * 100) if self.total_enemies > 0 else 0

        # Draw metrics
        metrics_font = pygame.font.Font(None, 24)
        title_font = pygame.font.Font(None, 28)

        title = title_font.render("Performance", True, (0, 0, 0))
        self.screen.blit(title, (panel_x + 10, 20))

        y_pos = 60
        metrics = [
            ("Coverage:", f"{coverage:.1f}%"),
            ("", f"{explored_cells}/{total_cells} cells"),
            ("", ""),
            ("Detection Rate:", f"{detection_rate:.1f}%"),
            ("Enemies Found:", f"{self.visited_enemies}/{self.total_enemies}"),
            ("", ""),
            ("Mission Time:", f"{self.mission_time_before_self_destruct}"),
            ("", ""),
            ("Agent Position:", f"({agent_r}, {agent_c})")
        ]

        for label, value in metrics:
            if label:
                text = metrics_font.render(label, True, (0, 0, 0))
                self.screen.blit(text, (panel_x + 10, y_pos))
                y_pos += 25
            if value:
                value_text = metrics_font.render(value, True, (50, 50, 200))
                self.screen.blit(value_text, (panel_x + 20, y_pos))
                y_pos += 30

        pygame.display.update()  # Update the display
        # pygame.time.delay(50)   # Slow down the rendering

    def reset(self, *, seed=None, options=None):
        # --- Generate a new map ------------------------------------------------------------------------
        self.map = self.generate_map(rows=40, cols=40, num_enemies=20)

        # --- Reinitialize dependent properties ---------------------------------------------------------
        self.num_rows, self.num_cols = self.map.shape
        self.seen_map = np.zeros((self.num_rows, self.num_cols), dtype=bool)
        self.start_position = tuple(np.argwhere(self.map == 'S')[0])
        self.state = self.start_position
        self.mission_time_before_self_destruct = 1000

        self.total_enemies = np.sum(self.map == '#')
        self.visited_enemies = 0

        # --- Update Pygame screen if dimensions changed ------------------------------------------------
        self.screen = pygame.display.set_mode(
            (self.num_cols * self.cell_size + self.info_panel_width, self.num_rows * self.cell_size)
        )

        info = {}
        return self.get_RL_agent_local_observation(), info
    
if __name__ == "__main__":
    # Understanding the state and action spaces used in the Outer Rim RL Environment
    env = OuterRimEnv()
    print(env.observation_space)
    print(env.action_space)
    print(env.reset())



    env = OuterRimEnv()

    episodes = 1
    for episode in range(1, episodes+1):
        obs, _ = env.reset()
        done = False
        episode_score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_score += reward

        print(f"Episode: {episode} Score: {episode_score} Enemies found: {info['visited_enemies']}/{info['total_enemies']}")

    env.close()

    log_path = os.path.join('Training_Star_Wars_Galaxy_Phase_1', 'logs')
    print(log_path)

    PPO_DRL_model = PPO(
        'MultiInputPolicy',
        env,
        verbose=1,
        tensorboard_log=log_path,
        batch_size=1024,
        n_steps=4096,
        learning_rate=3e-4,
        n_epochs=15,
        clip_range=0.2,
        ent_coef=0.01,
        gae_lambda=0.95,
        vf_coef=0.5,
    )

    PPO_Model_Custom = os.path.join('Training_Star_Wars_Galaxy_Phase_1', 'Saved RL Models', 'PPO_Model_Star_Wars_Galaxy_1M')
    reloaded_PPO_DRL_model = PPO.load(PPO_Model_Custom, env=env)

    env = OuterRimEnv()

    episodes = 5
    total_enemies_found = 0

    for episode in range(1, episodes+1):
        obs, _ = env.reset()
        print(f"Initial State: {obs}")
        done = False
        episode_score = 0

        while not done:
            env.render()
            action, _ = reloaded_PPO_DRL_model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_score += reward

        total_enemies_found += info['visited_enemies']
        print(f"Episode: {episode} Score: {episode_score} Enemies found: {info['visited_enemies']}/{info['total_enemies']}")

    env.close()

    average_enemies_found = total_enemies_found / episodes
    print(f"\nAverage enemies found over {episodes} episodes: {average_enemies_found:.2f}")