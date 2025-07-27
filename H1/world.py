import numpy as np
import random
import config

class World:
    def __init__(self):
        self.width  = config.WORLD_WIDTH
        self.height = config.WORLD_HEIGHT
        self.shift_interval = random.randint(config.SHIFT_INTERVAL_MIN,
                                             config.SHIFT_INTERVAL_MAX)
        self.generations_since_shift = 0
        self.regenerate()

    def clone(self):
        """
        Return a deep copy of the world so that each agent can be evaluated
        in an identical environment (prevents food depletion bias).
        """
        new = World.__new__(World)       # bypass __init__
        new.width  = self.width
        new.height = self.height
        new.grid   = self.grid.copy()    # deep copy of maze
        new.food   = self.food.copy()    # copy of food set
        # copy meta-state (not strictly needed for evaluation)
        new.shift_interval        = self.shift_interval
        new.generations_since_shift = self.generations_since_shift
        return new

    def regenerate(self):
        self.generate_maze()
        self.spawn_food()
        self.generations_since_shift = 0
        self.shift_interval = random.randint(config.SHIFT_INTERVAL_MIN,
                                             config.SHIFT_INTERVAL_MAX)
        self.just_shifted = True

    def step_generation(self):

        self.generations_since_shift += 1
        if self.generations_since_shift >= self.shift_interval:
            self.regenerate()
        else:
            self.just_shifted = False

    def generate_maze(self):
        rows = (self.height - 1) // 2
        cols = (self.width  - 1) // 2
        # Start with all walls
        self.grid = np.ones((self.height, self.width), dtype=int)
        visited = [[False] * cols for _ in range(rows)]

        def cell_to_grid(r, c):
            return 1 + 2*r, 1 + 2*c

        def dfs(r, c):
            visited[r][c] = True
            gy, gx = cell_to_grid(r, c)
            self.grid[gy, gx] = 0
            dirs = [(0,1), (0,-1), (1,0), (-1,0)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    wy, wx = gy + dr, gx + dc
                    self.grid[wy, wx] = 0
                    dfs(nr, nc)

        dfs(0, 0)

        if config.MAZE_REMOVE_FRACTION > 0:
            wall_coords = np.argwhere(self.grid == 1)
            n_remove = int(len(wall_coords) * config.MAZE_REMOVE_FRACTION)
            if n_remove > 0:
                remove_idx = np.random.choice(len(wall_coords),
                                              n_remove, replace=False)
                for y, x in wall_coords[remove_idx]:
                    self.grid[y, x] = 0

    def spawn_food(self):
        self.food = set()
        for _ in range(config.NUM_FOOD_CLUSTERS):
            # Choose cluster center
            cx = random.randrange(self.width)
            cy = random.randrange(self.height)
            placed = 0
            while placed < config.FOOD_PER_CLUSTER:
                # Random point near center
                x = cx + random.randint(-3, 3)
                y = cy + random.randint(-3, 3)
                if (0 <= x < self.width and 0 <= y < self.height
                        and self.grid[y, x] == 0
                        and (x, y) not in self.food):
                    self.food.add((x, y))
                    placed += 1

    def is_free(self, x, y):
        return (0 <= x < self.width
                and 0 <= y < self.height
                and self.grid[y, x] == 0)
