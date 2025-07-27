import random
import numpy as np

import config
from creature import Forager, THETA_LEN
from learning import LearningModule
from neural_net import Brain

# ---------- GA helpers (unchanged) ------------------------------------------
def crossover(g1, g2):
    mask = np.random.rand(*g1.shape) < config.CROSSOVER_RATE
    return np.where(mask, g1, g2)

def mutate(g):
    mask  = np.random.rand(*g.shape) < config.MUTATION_RATE
    noise = np.random.randn(*g.shape) * config.MUTATION_STD
    return g + mask * noise

def select_parents(pop, fits):
    k = max(2, int(len(pop) * config.ELITISM_RATE))
    return [pop[i] for i in np.argsort(fits)[-k:]]
# ----------------------------------------------------------------------------

def breed(parents, target_size):
    kids = []
    while len(kids) < target_size:
        p1, p2 = random.sample(parents, 2)
        genome_child = mutate(crossover(p1.genome, p2.genome))
        x0, y0 = random.randrange(config.WORLD_WIDTH), random.randrange(config.WORLD_HEIGHT)
        child = Forager(x0, y0)
        child.set_genome(genome_child)
        kids.append(child)
    return kids

# ----------------------------------------------------------------------------
def run_generation(population, world):
    """
    Evaluate each agent in its **own cloned world** so food removal by one
    individual does not disadvantage the next.
    """
    fitnesses = []

    for parent in population:
        # clone parent genome into fresh agent
        x0, y0 = random.randrange(config.WORLD_WIDTH), random.randrange(config.WORLD_HEIGHT)
        agent = Forager(x0, y0)
        agent.set_genome(parent.genome.copy())
        agent.reset(x0, y0)

        # --- use a deep copy of the world ---
        wcopy = world.clone()

        learner = LearningModule()
        tot = 0.0
        for _ in range(config.TICKS_PER_EPISODE):
            tot += agent.step(wcopy, learner)   # note wcopy
        fitnesses.append(tot)

    parents  = select_parents(population, fitnesses)
    children = breed(parents, len(population) - len(parents))
    next_pop = parents + children

    return next_pop, {
        'best_fitness': float(np.max(fitnesses)),
        'avg_fitness':  float(np.mean(fitnesses)),
    }

def evaluate_population(population, world):
    """
    Returns average fitness of current population in the given world
    (cloned per agent).  No genomes are modified.
    """
    scores = []
    learner = LearningModule()
    for parent in population:
        # clone world for fairness
        wcopy = world.clone()
        # fresh copy of agent
        agent = Forager(parent.x, parent.y)
        agent.set_genome(parent.genome.copy())
        agent.reset(agent.x, agent.y)

        tot = 0.0
        for _ in range(config.TICKS_PER_EPISODE):
            tot += agent.step(wcopy, learner)
        scores.append(tot)
    return float(np.mean(scores))
