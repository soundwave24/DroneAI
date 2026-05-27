import random
from npc_ships import SeperatistShip, RepublicShip

class EntityManager:
    def __init__(self, world, n_rep=3, n_sep=3):
        self.world = world
        self.n_rep, self.n_sep = n_rep, n_sep
        self.spawn()

    def spawn(self):
        rows, cols = self.world.num_rows, self.world.num_cols
        border_cells = [(0,c) for c in range(cols)] + [(rows-1,c) for c in range(cols)] + \
                       [(r,0) for r in range(rows)] + [(r,cols-1) for r in range(rows)]
        random.shuffle(border_cells)
        for _ in range(self.n_rep):
            self.world.entities.append(RepublicShip(border_cells.pop(), vision_radius=5, move_every=2))
        for _ in range(self.n_sep):
            self.world.entities.append(SeperatistShip(border_cells.pop(), vision_radius=3, move_every=2))
