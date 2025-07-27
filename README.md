# Evolving Plasticity in Dynamic Environments

---

## Project Overview

This repository contains a full simulation framework for studying how neural plasticity (within-lifetime learning) co-evolves with genetically encoded behavior in non-stationary environments. Agents (“foragers”) live in a procedurally generated 2D maze, collect food, and evolve over hundreds of generations. Crucially, each agent can also learn during its lifetime via local weight updates.

By varying how often the maze reshuffles, the code tests three hypotheses derived from the Baldwin effect:

- **H1**: Plastic agents recover fitness faster than fixed-weight agents after environment shifts.
- **H2**: Frequent shifts select for higher plasticity (higher learning-rate vectors and eligibility decay).
- **H3**: Rare shifts promote genetic assimilation: the drift between inherited and learned weights collapses over time.

The code base is self-contained (pure Python + PyTorch + NumPy + Pygame) and produces logs and plots for all key metrics: fitness, plasticity parameters, drift, post-shift AUC/slope.

---

## Key Features

- **Neuroevolution with plasticity**: Genomes encode both initial neural weights (θ₀) and learning meta-parameters (η, λ, hebb).
- **Lifetime learning**: Agents update their neural network weights online during a 1,000-tick episode using either a TD-like rule with eligibility traces or a Hebbian-style rule.
- **Dynamic environment**: Procedural mazes and clustered food distributions; full regeneration at configurable intervals (rare, normal, frequent).
- **Comprehensive logging**: Per-generation CSV logs for fitness, plasticity metrics, and drift; separate post-shift logs for recovery analysis.
- **Reproducible plots**: Scripts and notebooks (if included) to produce fitness curves, AUC/slope by shift, plasticity trends, and drift over generations.
- **Interactive demo**: A simple visualization (Pygame) to watch agents forage at selected checkpoints.

---

## Repository Structure

- `config.py` — Global hyperparameters (world size, GA rates, NN sizes, etc.)
- `creature.py` — Agent (Forager) definition, genome decoding, sensing, acting
- `neural_net.py` — PyTorch implementation of the agent’s brain (two-layer MLP)
- `learning.py` — Lifetime learning module (TD-like or Hebbian updates with eligibility traces)
- `world.py` — Maze generation, food spawning, shift scheduling
- `evolve.py` — Evolutionary loop per generation (selection, crossover, mutation, metric logging)
- `main.py` — Entry point: runs full experiments, logging, saving genomes, optional demos
- `visualize.py` — Pygame visualization for agent behavior within a snapshot world
- `utils.py` — Helper utilities (selection helpers, distance computations, etc.)

**Folders:**
- `logs/` — Generated CSV logs (`stats_log.csv`, `postshift_log.csv`)
- `plots/` — Generated figures for H1–H3 

---


## Getting Started

1. Clone the repository.
2. *(Optional but recommended)* Create a virtual environment.
3. Install dependencies.
4. Run the main experiment:
   ```bash
   python main.py
   ```

### Program Prompts

- **Shift profile**: `rare`, `frequent`, or `normal`
  - `rare` → reshuffle every 15–20 generations  
  - `frequent` → reshuffle every 1–3 generations  
- **Generation checkpoints**: comma-separated list (e.g., `1,5,10`) to pause and optionally visualize.

### Outputs

- `logs/stats_log.csv`: Per-generation stats (fitness, median η/λ, drift)
- `logs/postshift_log.csv`: Post-shift fitness metrics over 10-generation recovery windows
- `saved_genomes/`: Genomes saved every `SAVE_INTERVAL` generations (default = 10)

---

## Experimental Profiles

### Reproduce H1 (Plastic vs Fixed)

- Run default (plastic agents)
- For fixed agents: disable learning updates or clamp `η = 0` in `set_genome()`

### Reproduce H2 and H3 (Rare vs Frequent)

- Run `frequent` profile (1–3 generations)
- Run `rare` profile (15–20 generations)
- Compare median `η`, `λ`, and drift metrics

To ensure repeatability, manually seed Python/NumPy/PyTorch RNGs.

---

## Outputs and Metrics

**`stats_log.csv` Columns:**
- `generation` — generation number
- `best_fitness` — max food collected
- `avg_fitness` — average fitness
- `median_eta` — median learning rate (η)
- `median_lambda` — median eligibility decay (λ)
- `median_drift` — median L2 norm of `(θ_final − θ₀)`

**`postshift_log.csv` Columns:**
- `shift_id` — environment shift index
- `generation` — time since run began
- `best_fitness`, `avg_fitness` — post-shift recovery data

You can compute AUC and slope from `postshift_log.csv` using 10-gen recovery slices.

---

## Genome Encoding

Genome layout (`1D NumPy array`):

```
[ θ₀ | η | λ | hebb ]
```

- **θ₀ length** = `HIDDEN_SIZE * NUM_SENSORS + HIDDEN_SIZE + NUM_OUTPUTS * HIDDEN_SIZE + NUM_OUTPUTS`
- **η** — same length as θ₀ (per-parameter learning rate)
- **λ** — single float [0,1]
- **hebb** — integer flag (0 or 1)

The method `creature.set_genome()` includes bounds clipping to maintain numerical stability.

---


## How Learning Works During Life

1. Agent senses environment (9D vector)
2. Neural net outputs Q-like scores; argmax selected
3. Agent acts, collects reward, observes next state
4. `learning_module.learn()` updates weights based on η, λ, traces
5. Only phenotype (current net) is modified; genome stays intact

---

## Evolutionary Loop (`run_generation` in `evolve.py`)

1. Clone parent genome into fresh agent
2. Simulate episode; record reward (fitness)
3. Compute drift: `||θ_final − θ₀||`
4. Select elites, breed/mutate children
5. Log medians for η, λ, drift

---


## Extensions and Future Directions

- Multi-agent dynamics (competition/cooperation)
- More complex tasks (navigation, memory, control)
- Meta-learning integration (MAML)
- Richer plasticity rules (neuromodulation, synaptic tags)
- Hierarchical/recurrent networks (temporal abstraction)

---


## License

MIT License – 2025 Alpar Arman

Feel free to use, modify, and distribute – a citation or star is always appreciated!
