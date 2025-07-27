# config.py
# Global configuration for the Baldwinian gridworld experiment

# --- Environment parameters ---
WORLD_WIDTH = 80         # number of columns in the grid
WORLD_HEIGHT = 60        # number of rows in the grid

NUM_FOOD_CLUSTERS = 5    # how many clusters of food to spawn each environment
FOOD_PER_CLUSTER = 10    # number of food items in each cluster

MAZE_REMOVE_FRACTION = 0.70
# Environment will regenerate (maze + food) every K generations,
# where K is sampled uniformly between SHIFT_INTERVAL_MIN and SHIFT_INTERVAL_MAX.
SHIFT_INTERVAL_MIN = 5
SHIFT_INTERVAL_MAX = 10

# --- Genetic Algorithm parameters ---
POPULATION_SIZE = 128    # total number of agents per generation
ELITISM_RATE   = 0.5     # fraction of top performers selected as parents
CROSSOVER_RATE = 0.5     # probability of performing crossover vs. copying parent genome
MUTATION_RATE  = 0.2     # per-gene mutation probability
MUTATION_STD   = 0.1     # standard deviation of Gaussian noise for mutation

# --- Evolution run parameters ---
NUM_GENERATIONS = 500    # total number of generations to run
SAVE_INTERVAL   = 10     # save genomes every N generations

# --- Genome structure ---
# Genome is a flat vector composed of:
# [ theta_0 | eta | lambda | hebb_toggle ]
#  - theta_0: initial network weights
#  - eta: per-weight learning rates (same length as theta_0)
#  - lambda: eligibility trace decay coefficient (scalar)
#  - hebb_toggle: binary flag (0=TD(lambda), 1=Hebbian)

# --- Neural network architecture ---
NUM_SENSORS = 9         # number of inputs to the value network (e.g. local observations)
HIDDEN_SIZE = 64        # hidden-layer size for the MLP
NUM_OUTPUTS = 4         # number of discrete actions (up/down/left/right)

# --- Learning (TD(lambda)) parameters ---
GAMMA = 0.99            # discount factor for temporal-difference updates

# --- Simulation settings ---
TICKS_PER_EPISODE     = 1000  # max steps per agent lifetime evaluation
NUM_EPISODES_PER_EVAL = 1     # how many episodes per generation for fitness evaluation

# --- Logging and saving directories ---
LOG_DIR    = "logs"
PLOTS_DIR  = "plots"
GENOME_DIR = "saved_genomes"

# --- Baseline configuration ---
# Settings for PPO-LSTM baseline using Stable-Baselines3
BASELINE_ALGO            = "PPO"
BASELINE_TOTAL_TIMESTEPS = 1_000_000
