import math
import torch
import numpy as np
import config
from neural_net import Brain

THETA_LEN = (
    config.HIDDEN_SIZE * config.NUM_SENSORS +
    config.HIDDEN_SIZE +
    config.NUM_OUTPUTS * config.HIDDEN_SIZE +
    config.NUM_OUTPUTS
)

class Creature:

    def __init__(self, x, y, genome=None):
        self.x, self.y = x, y

        if genome is None:
            theta0 = np.random.randn(THETA_LEN)
            eta    = np.zeros(THETA_LEN)
            lam    = np.random.rand()
            hebb   = np.random.randint(0, 2)
            genome = np.concatenate([theta0, eta, [lam, hebb]])

        self.set_genome(genome)
        self.e_trace = np.zeros(THETA_LEN)
        self.total_reward = 0.0

    def step(self, world, learning_module):
        """
        One time-step loop:
          1. Observe
          2. Choose & execute action
          3. Receive reward
          4. Call learning rule
        Returns the reward obtained this step.
        """
        # 1. Observation before action
        prev_obs = self.sense(world)

        # 2. Action
        action = self.act(prev_obs)
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[action]
        nx, ny = self.x + dx, self.y + dy
        if world.is_free(nx, ny):
            self.x, self.y = nx, ny

        # 3. Reward
        reward = 0
        if (self.x, self.y) in world.food:
            reward = 1
            world.food.remove((self.x, self.y))
        self.total_reward += reward

        # 4. Observation after action
        new_obs = self.sense(world)

        # 5. Plasticity update
        learning_module.learn(self, prev_obs, action, reward, new_obs)
        return reward


    def _decode_genome(self, genome):
        theta0 = genome[:THETA_LEN]
        eta    = genome[THETA_LEN:2*THETA_LEN]
        lam    = float(genome[2*THETA_LEN])
        hebb   = int(genome[2*THETA_LEN + 1])
        return theta0, eta, lam, hebb

    def set_genome(self, g):
        """Assign new genome, refresh decoded fields and rebuild brain once."""
        self.genome = g
        self.theta0, self.eta, self.lmbda, self.hebb = self._decode_genome(g)
        self.brain = Brain(self.theta0)

    def reset(self, x, y):
        self.x, self.y = x, y
        self.total_reward = 0.0
        self.e_trace[:] = 0
        self.brain = Brain(self.theta0)

    def sense(self, world):
        """
        9-element feature vector:
          0: dx to nearest food  (normalised −1…1)
          1: dy to nearest food  (normalised −1…1)
          2: distance to food    (0…1)
          3–6: wall_up/down/left/right (0/1)
          7: bias (constant 1)
          8: Gaussian noise (N(0,0.1))
        """
        # find nearest food (Manhattan distance)
        if world.food:
            tgt = min(
                world.food,
                key=lambda f: abs(f[0] - self.x) + abs(f[1] - self.y)
            )
            dx = tgt[0] - self.x
            dy = tgt[1] - self.y
            dist = abs(dx) + abs(dy)
        else:
            dx = dy = 0
            dist = (world.width + world.height)  # max
        # normalisation
        dx_norm = dx / world.width
        dy_norm = dy / world.height
        dist_norm = dist / (world.width + world.height)

        # wall detectors
        wall_up    = int(not world.is_free(self.x, self.y - 1))
        wall_down  = int(not world.is_free(self.x, self.y + 1))
        wall_left  = int(not world.is_free(self.x - 1, self.y))
        wall_right = int(not world.is_free(self.x + 1, self.y))

        bias  = 1.0
        noise = np.random.randn() * 0.1

        return np.array([
            dx_norm, dy_norm, dist_norm,
            wall_up, wall_down, wall_left, wall_right,
            bias, noise
        ], dtype=np.float32)

    def act(self, obs):
        out = self.brain(obs)
        return int(torch.argmax(out).item())

class Forager(Creature):
    """Specialised agent (no extra behaviour yet)."""
    pass
