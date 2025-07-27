import numpy as np, random, config
class World:
    def __init__(self, shift_min=None, shift_max=None):
        self.width = config.WORLD_WIDTH
        self.height = config.WORLD_HEIGHT
        self.min_int = shift_min if shift_min is not None else config.SHIFT_INTERVAL_MIN
        self.max_int = shift_max if shift_max is not None else config.SHIFT_INTERVAL_MAX
        self.shift_interval = random.randint(self.min_int, self.max_int)
        self.generations_since_shift = 0
        self.regenerate()
    def clone(self):
        new = World.__new__(World)
        new.width, new.height = self.width, self.height
        new.grid = self.grid.copy()
        new.food = self.food.copy()
        new.min_int, new.max_int = self.min_int, self.max_int
        new.shift_interval = self.shift_interval
        new.generations_since_shift = self.generations_since_shift
        new.just_shifted = self.just_shifted
        return new
    def regenerate(self):
        self.generate_maze()
        self.spawn_food()
        self.generations_since_shift = 0
        self.shift_interval = random.randint(self.min_int, self.max_int)
        self.just_shifted = True
    def step_generation(self):
        self.generations_since_shift += 1
        if self.generations_since_shift >= self.shift_interval:
            self.regenerate()
        else:
            self.just_shifted = False
    def generate_maze(self):
        rows = (self.height-1)//2
        cols = (self.width-1)//2
        self.grid = np.ones((self.height, self.width), dtype=int)
        visited = [[False]*cols for _ in range(rows)]
        def cell_to_grid(r,c): return 1+2*r,1+2*c
        def dfs(r,c):
            visited[r][c]=True
            gy,gx=cell_to_grid(r,c)
            self.grid[gy,gx]=0
            dirs=[(0,1),(0,-1),(1,0),(-1,0)]
            random.shuffle(dirs)
            for dr,dc in dirs:
                nr,nc=r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]:
                    wy,wx=gy+dr,gx+dc
                    self.grid[wy,wx]=0
                    dfs(nr,nc)
        dfs(0,0)
        if config.MAZE_REMOVE_FRACTION>0:
            walls=np.argwhere(self.grid==1)
            n_remove=int(len(walls)*config.MAZE_REMOVE_FRACTION)
            if n_remove>0:
                idx=np.random.choice(len(walls),n_remove,replace=False)
                for y,x in walls[idx]:
                    self.grid[y,x]=0
    def spawn_food(self):
        self.food=set()
        for _ in range(config.NUM_FOOD_CLUSTERS):
            cx=random.randrange(self.width)
            cy=random.randrange(self.height)
            placed=0
            while placed<config.FOOD_PER_CLUSTER:
                x=cx+random.randint(-3,3)
                y=cy+random.randint(-3,3)
                if 0<=x<self.width and 0<=y<self.height and self.grid[y,x]==0 and (x,y) not in self.food:
                    self.food.add((x,y))
                    placed+=1
    def is_free(self,x,y):
        return 0<=x<self.width and 0<=y<self.height and self.grid[y,x]==0
