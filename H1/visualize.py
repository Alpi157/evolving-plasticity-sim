# visualize.py
import pygame
import config
from world import World
from creature import Forager

# --- simple colors ----------------------------------------------------------
WALL   = (60, 60, 60)
FOOD   = (0, 180, 0)
AGENT  = (0, 120, 255)
BLACK  = (0, 0, 0)

TILE   = 8          # pixel size of each grid cell
FPS    = 30         # animation speed
STEPS  = 300        # how long to run the demo

# ---------------------------------------------------------------------------
def draw(window: pygame.Surface, world: World, agents: list[Forager]):
    window.fill(BLACK)
    # walls
    for y in range(world.height):
        for x in range(world.width):
            if world.grid[y, x] == 1:
                pygame.draw.rect(
                    window, WALL,
                    (x * TILE, y * TILE, TILE, TILE)
                )
    # food
    for (fx, fy) in world.food:
        pygame.draw.circle(
            window, FOOD,
            (fx * TILE + TILE // 2, fy * TILE + TILE // 2),
            TILE // 4
        )
    # agents
    for a in agents:
        pygame.draw.circle(
            window, AGENT,
            (a.x * TILE + TILE // 2, a.y * TILE + TILE // 2),
            TILE // 3
        )
    pygame.display.flip()

# ---------------------------------------------------------------------------
def run_demo(population: list[Forager], world: World):
    pygame.init()
    win = pygame.display.set_mode(
        (world.width * TILE, world.height * TILE)
    )
    clock = pygame.time.Clock()

    # clone world so demo doesn’t alter main sim
    wcopy = world.clone()

    # deep-copy agents so they don’t alter GA population
    demo_agents = [
        Forager(a.x, a.y, genome=a.genome.copy()) for a in population
    ]

    for _ in range(STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        for agent in demo_agents:
            # no learning during demo – pass a dummy learner
            agent.step(wcopy, learning_module=DummyLearner())
        draw(win, wcopy, demo_agents)
        clock.tick(FPS)
    pygame.quit()

# dummy learning module that does nothing (for visual playback)
class DummyLearner:
    def learn(self, *args, **kwargs):
        pass
