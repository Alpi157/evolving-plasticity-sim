import math, random

def find_nearest(agent, others):
    best, bd = None, math.inf
    for o in others:
        d = abs(o.x - agent.x) + abs(o.y - agent.y)
        if d < bd:
            best, bd = o, d
    return best, bd

def tournament_select(pop, fits, k):
    best, bf = None, -math.inf
    for _ in range(k):
        i = random.randrange(len(pop))
        if fits[i] > bf:
            best, bf = pop[i], fits[i]
    return best
