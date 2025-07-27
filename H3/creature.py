import torch, numpy as np, config
from neural_net import Brain

THETA_LEN = config.HIDDEN_SIZE * config.NUM_SENSORS + config.HIDDEN_SIZE + config.NUM_OUTPUTS * config.HIDDEN_SIZE + config.NUM_OUTPUTS

class Creature:
    def __init__(self, x, y, genome=None):
        self.x, self.y = x, y
        if genome is None:
            theta0 = np.random.randn(THETA_LEN)
            eta = np.abs(np.random.randn(THETA_LEN))
            lam = np.random.rand()
            hebb = np.random.randint(0, 2)
            genome = np.concatenate([theta0, eta, [lam, hebb]])
        self.set_genome(genome)
        self.e_trace = np.zeros(THETA_LEN)
        self.total_reward = 0.0
    def step(self, world, learning_module):
        prev_obs = self.sense(world)
        action = self.act(prev_obs)
        dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[action]
        nx, ny = self.x + dx, self.y + dy
        if world.is_free(nx, ny):
            self.x, self.y = nx, ny
        reward = 0
        if (self.x, self.y) in world.food:
            reward = 1
            world.food.remove((self.x, self.y))
        self.total_reward += reward
        new_obs = self.sense(world)
        learning_module.learn(self, prev_obs, action, reward, new_obs)
        return reward
    def _decode_genome(self, g):
        theta0 = g[:THETA_LEN]
        eta = g[THETA_LEN:2*THETA_LEN]
        lam = float(g[2*THETA_LEN])
        hebb = int(g[2*THETA_LEN+1])
        return theta0, eta, lam, hebb

    def set_genome(self, g):
        """
        Decode genome → (theta0, eta, lambda, hebb),
        then **clip meta-params** to valid ranges and rebuild the brain.
        """
        self.genome = g

        theta0 = g[:THETA_LEN]
        eta = g[THETA_LEN:2 * THETA_LEN]
        lam = float(g[2 * THETA_LEN])
        hebb = int(g[2 * THETA_LEN + 1])

        # --- enforce bounds --------------------------------------------
        eta = np.clip(eta, 0.0, config.ETA_MAX)
        lam = float(np.clip(lam, 0.0, 1.0))

        # write clipped values back into the genome so it stays consistent
        self.genome[THETA_LEN:2 * THETA_LEN] = eta
        self.genome[2 * THETA_LEN] = lam

        # store fields
        self.theta0 = theta0.copy()
        self.eta = eta
        self.lmbda = lam
        self.hebb = hebb

        # rebuild neural net
        self.brain = Brain(self.theta0)
    def reset(self, x, y):
        self.x, self.y = x, y
        self.total_reward = 0.0
        self.e_trace[:] = 0
        self.brain = Brain(self.theta0)
    def sense(self, world):
        if world.food:
            tgt = min(world.food, key=lambda f: abs(f[0]-self.x)+abs(f[1]-self.y))
            dx, dy = tgt[0]-self.x, tgt[1]-self.y
            dist = abs(dx)+abs(dy)
        else:
            dx = dy = 0
            dist = world.width + world.height
        dx_norm = dx / world.width
        dy_norm = dy / world.height
        dist_norm = dist / (world.width + world.height)
        wall_up = int(not world.is_free(self.x, self.y-1))
        wall_down = int(not world.is_free(self.x, self.y+1))
        wall_left = int(not world.is_free(self.x-1, self.y))
        wall_right = int(not world.is_free(self.x+1, self.y))
        bias = 1.0
        noise = np.random.randn()*0.1
        return np.array([dx_norm, dy_norm, dist_norm, wall_up, wall_down, wall_left, wall_right, bias, noise], dtype=np.float32)
    def act(self, obs):
        out = self.brain(obs)
        return int(torch.argmax(out).item())
class Forager(Creature):
    pass
