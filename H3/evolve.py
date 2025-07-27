import random, numpy as np, config
from creature import Forager, THETA_LEN
from learning  import LearningModule, CLIP_DRIFT

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

def breed(parents, target_size):
    kids = []
    while len(kids) < target_size:
        p1, p2 = random.sample(parents, 2)
        g_child = mutate(crossover(p1.genome, p2.genome))
        x0, y0  = random.randrange(config.WORLD_WIDTH), random.randrange(config.WORLD_HEIGHT)
        child   = Forager(x0, y0)
        child.set_genome(g_child)
        kids.append(child)
    return kids

def _flatten_params(brain):
    return np.concatenate([p.detach().numpy().ravel() for p in brain.parameters()])

def run_generation(population, world):
    fitnesses, drifts = [], []

    for parent in population:
        # fresh copy of parent
        x0, y0 = random.randrange(config.WORLD_WIDTH), random.randrange(config.WORLD_HEIGHT)
        agent  = Forager(x0, y0)
        agent.set_genome(parent.genome.copy())
        agent.reset(x0, y0)

        theta_inherited = agent.theta0.copy()        # snapshot before lifetime
        wcopy  = world.clone()
        learner = LearningModule()

        tot = 0.0
        for _ in range(config.TICKS_PER_EPISODE):
            tot += agent.step(wcopy, learner)

        fitnesses.append(tot)

        theta_final = _flatten_params(agent.brain)
        d = np.linalg.norm(theta_final - theta_inherited)
        drifts.append(d if d < CLIP_DRIFT and not np.isnan(d) else np.nan)

    parents  = select_parents(population, fitnesses)
    children = breed(parents, len(population) - len(parents))
    next_pop = parents + children

    eta_med   = float(np.median([np.median(a.eta) for a in next_pop]))
    lam_med   = float(np.median([a.lmbda for a in next_pop]))
    drift_med = float(np.nanmedian(drifts))

    return next_pop, {
        "best_fitness"  : float(np.max(fitnesses)),
        "avg_fitness"   : float(np.mean(fitnesses)),
        "median_eta"    : eta_med,
        "median_lambda" : lam_med,
        "median_drift"  : drift_med
    }
