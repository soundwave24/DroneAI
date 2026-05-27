# NOTE: I introduced the idea of stochasticity in the chasing mechanism of the 
# 'SeparatistShip' and 'RepublciShip'. Because if they just go one step towards each other,
# the behaviour would be the chase will just be in a straight line/1 dimension.

# Here's is how I introduced stochasticity:
# - When a 'SeparatistShip' sees a 'RepublicShip' in view, it will have:
#   -> 85% chance of chasing it
#   -> 5% chance to take any of the other 3 actions
# - When a 'RepublicShip' sees a 'SeparatistShip' in view,  it will have:
#   -> 60% → move directly away from threat
#   -> 20% → move perpendicular (left or right relative to escape direction)
#   -> 0% → never move toward the threat (suicide)
#   -> If direct move is invalid (blocked/wall):
#      => Redistribute to 50% left / 50% right
#   -> If cornered (only one escape direction), always pick valid escape

import random
import numpy as np
from mobile_entity import MobileEntity, DIRS

# -------------------------------------------------------
# Hostile to RL agent
class SeperatistShip(MobileEntity):
    def __init__(self, pos, vision_radius=3, move_every=2):
        super().__init__(pos, vision_radius)
        self.move_every = move_every      # move only every N steps
        self._tick = 0                    # internal counter

    def choose_action_stochastic(self, world):
        self._tick += 1
        if self._tick % self.move_every != 0:
            return (0, 0)  # Skip movement this turn
        
        targets = self.sense(world, mask_types=(world.agent.__class__, RepublicShip))
        if not targets:
            return super().choose_action(world)  # random patrol

        rand = random.random()
        if rand <= 0.85:
            target, _ = random.choice(targets)
            return self._step_towards(target.pos)
        else:
            # Pick randomly from the 3 *other* directions
            intended = self._step_towards(random.choice(targets)[0].pos)
            other_dirs = [d for d in DIRS if d != intended]
            return random.choice(other_dirs)


    # ///////////////////////////////////////////////////////////////////

    
    def choose_action_deterministic(self, world):
        self._tick += 1
        if self._tick % self.move_every != 0:
            return (0, 0)  # Skip movement this turn
        
        targets = self.sense(world, mask_types=(world.agent.__class__, RepublicShip))
        if not targets:
            return super().choose_action(world)        # random patrol
        target, _ = random.choice(targets)             # pick random visible target
        return self._step_towards(target.pos)
    
    
    def _step_towards(self, goal):
        rr, cc = self.pos
        gr, gc = goal
        dr = np.sign(gr - rr)
        dc = np.sign(gc - cc)

        # Force non-zero move: prioritize a valid direction
        if dr != 0 and dc != 0:
            if random.random() < 0.5:
                dc = 0
            else:
                dr = 0
        elif dr == 0 and dc == 0:
            # Already at goal — choose random direction to move
            dr, dc = random.choice(DIRS)
        
        return (dr, dc)


# -------------------------------------------------------
# Neutral to RL agent
class RepublicShip(MobileEntity):
    def __init__(self, pos, vision_radius=5, move_every=1):
        super().__init__(pos, vision_radius)
        self.move_every = move_every
        self._tick = 0

    def choose_action_stochastic(self, world):
        self._tick += 1
        if self._tick % self.move_every != 0:
            return (0, 0)  # Stay in place this turn
        
        threats = self.sense(world, mask_types=(SeperatistShip,))
        if not threats:
            return super().choose_action(world)  # default random patrol

        # Step 1: Find closest threat
        min_dist = min(dist for _, dist in threats)
        closest_threats = [e for e, dist in threats if dist == min_dist]
        threat = random.choice(closest_threats)
        rr, cc = self.pos
        tr, tc = threat.pos

        # Step 2: Determine escape direction
        dr = -np.sign(tr - rr)
        dc = -np.sign(tc - cc)
        escape = (dr, dc)

        # Normalize to 1D escape (so it's a single direction per step)
        if dr != 0 and dc != 0:
            if random.random() < 0.5:
                dc = 0
            else:
                dr = 0
        escape = (dr, dc)

        # Step 3: Get perpendiculars (left/right relative to escape)
        if escape == (-1, 0):  # up
            perpendiculars = [(0, -1), (0, 1)]  # left/right
        elif escape == (1, 0):  # down
            perpendiculars = [(0, 1), (0, -1)]
        elif escape == (0, -1):  # left
            perpendiculars = [(1, 0), (-1, 0)]
        elif escape == (0, 1):  # right
            perpendiculars = [(-1, 0), (1, 0)]
        else:
            return super().choose_action(world)  # fallback safety

        # Step 4: Check valid moves
        moves = {}
        for move in [escape] + perpendiculars:
            new_pos = self.propose_move(move)
            if world.check_valid_position(new_pos):
                moves[move] = True
            else:
                moves[move] = False

        valid_perpendiculars = [m for m in perpendiculars if moves[m]]
        can_escape = moves[escape]

        # Step 5: Decide based on availability
        if can_escape:
            options = [escape] + valid_perpendiculars
            probs = [0.6] + [0.2 / len(valid_perpendiculars)] * len(valid_perpendiculars)
        elif valid_perpendiculars:
            options = valid_perpendiculars
            probs = [1.0 / len(valid_perpendiculars)] * len(valid_perpendiculars)
        else:
            return super().choose_action(world)  # stuck? random

        # Step 6: Sample action
        choice = random.choices(options, weights=probs, k=1)[0]
        return choice


    # //////////////////////////////////////////////////////////////


    def choose_action_deterministic(self, world):
        self._tick += 1
        if self._tick % self.move_every != 0:
            return (0, 0)  # Stay in place this turn
        
        threats = self.sense(world, mask_types=(SeperatistShip,))
        if not threats:
            return super().choose_action(world)  # random patrol

        # Compute minimum threat distance
        min_dist = min(dist for _, dist in threats)

        # Collect all threats at minimum distance
        closest_threats = [entity for entity, dist in threats if dist == min_dist]

        # Choose one randomly among the closest
        threat = random.choice(closest_threats)
        return self._step_away(threat.pos)
    

    def _step_away(self, threat_pos):
        rr, cc = self.pos
        tr, tc = threat_pos
        dr = -np.sign(tr - rr)
        dc = -np.sign(tc - cc)

        if dr != 0 and dc != 0:
            if random.random() < 0.5:
                dc = 0
            else:
                dr = 0
        elif dr == 0 and dc == 0:
            # Threat is in the same cell — move randomly to escape
            dr, dc = random.choice(DIRS)

        return (dr, dc)

