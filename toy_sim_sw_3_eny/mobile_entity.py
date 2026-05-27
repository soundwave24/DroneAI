import numpy as np
import random

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]  # N,S,W,E

class DummyAgent:
    def __init__(self, pos):
        self.pos = pos

class MobileEntity:
    def __init__(self, pos, vision_radius=3):
        self.pos = tuple(pos)
        self.vision_radius = vision_radius

    # ---------------------------------------------------
    # helpers
    # ---------------------------------------------------
    def sense(self, world, mask_types=None):
        """
        Returns list of (entity, dist) within vision_radius.
        If mask_types is given, only return those subclasses.
        """
        rr, cc = self.pos
        seen = []
        for e in world.entities:
            if e is self:
                continue
            if mask_types and not isinstance(e, mask_types):
                continue
            r, c = e.pos
            if abs(r-rr) <= self.vision_radius and abs(c-cc) <= self.vision_radius:
                seen.append((e, abs(r-rr)+abs(c-cc)))

        # Special check for the RL agent (if it's in the mask_types)
        if mask_types and (world.agent.__class__ in mask_types or any(
            issubclass(world.agent.__class__, t) for t in mask_types if isinstance(t, type))):
            
            r, c = world.state  # RL agent's position
            if abs(r - rr) <= self.vision_radius and abs(c - cc) <= self.vision_radius:
                fake_agent = DummyAgent((r, c))   # wrap it with a .pos
                seen.append((fake_agent, abs(r - rr) + abs(c - cc)))

        return seen

    # simple grid move with bounds-check done by Env
    def propose_move(self, drdc):
        r, c = self.pos
        dr, dc = drdc
        return (r+dr, c+dc)

    # default behaviour = random patrol
    def choose_action(self, world):
        return random.choice(DIRS)
