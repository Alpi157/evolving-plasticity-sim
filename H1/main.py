# main.py
import os, csv, random, numpy as np
import config
from creature import Forager, THETA_LEN
from evolve   import run_generation
from world    import World
from visualize import run_demo    # For manual playback

def main():
    # ----- user chooses demo generations -------------------------------
    chk_raw = input("Enter generation checkpoints (e.g. 1,5,10): ")
    checkpoints = sorted({
        int(x) for x in chk_raw.replace(" ", "").split(",") if x.isdigit()
    })
    # -------------------------------------------------------------------

    os.makedirs(config.LOG_DIR,    exist_ok=True)
    os.makedirs(config.PLOTS_DIR,  exist_ok=True)
    os.makedirs(config.GENOME_DIR, exist_ok=True)

    # initial population
    population = [
        Forager(random.randrange(config.WORLD_WIDTH),
                random.randrange(config.WORLD_HEIGHT))
        for _ in range(config.POPULATION_SIZE)
    ]

    world = World()  # persistent environment

    # ---- CSV log for normal generation stats ----
    with open(os.path.join(config.LOG_DIR, "stats_log.csv"), "w", newline="") as fcsv, \
         open(os.path.join(config.LOG_DIR, "postshift_log.csv"), "w", newline="") as fshift:

        stats_writer = csv.DictWriter(fcsv, fieldnames=["generation", "best_fitness", "avg_fitness"])
        shift_writer = csv.DictWriter(fshift, fieldnames=["shift_id", "generation", "best_fitness", "avg_fitness"])
        stats_writer.writeheader()
        shift_writer.writeheader()

        shift_mode = False
        postshift_counter = 0
        shift_id = 0

        for gen in range(1, config.NUM_GENERATIONS + 1):
            # Step the world forward (may or may not trigger a regeneration)
            prev_shift_state = world.generations_since_shift
            world.step_generation()
            did_shift = (world.generations_since_shift == 0 and prev_shift_state != 0)

            # Run a generation
            population, stats = run_generation(population, world)

            # Normal log
            stats_writer.writerow({
                "generation": gen,
                "best_fitness": stats["best_fitness"],
                "avg_fitness": stats["avg_fitness"]
            })

            print(f"Gen {gen:4d} | best: {stats['best_fitness']:.2f} | avg: {stats['avg_fitness']:.2f}")

            # === SHIFT DETECTION ===
            if did_shift:
                print(f"\n[ENV SHIFT] at Generation {gen} – starting post-shift adaptation log\n")
                shift_mode = True
                postshift_counter = 0
                shift_id += 1

            # === LOG POST-SHIFT EPISODES ===
            if shift_mode:
                shift_writer.writerow({
                    "shift_id": shift_id,
                    "generation": gen,
                    "best_fitness": stats["best_fitness"],
                    "avg_fitness": stats["avg_fitness"]
                })
                postshift_counter += 1
                if postshift_counter >= 10:
                    shift_mode = False

            # ------------- DEMO ---------------------------------------
            if gen in checkpoints:
                input(f"\nPress Enter to watch Generation {gen} ...")
                run_demo(population, world)
                print("Demo finished.\n")
            # ----------------------------------------------------------

            # save genomes
            if gen % config.SAVE_INTERVAL == 0 or gen == config.NUM_GENERATIONS:
                for i, ag in enumerate(population):
                    np.save(
                        os.path.join(
                            config.GENOME_DIR,
                            f"gen{gen:04d}_ind{i:03d}.npy"
                        ),
                        ag.genome
                    )

    print("Evolution complete. Logs saved to", config.LOG_DIR)

# ---------------- sanity check ---------------------------------------------
if __name__ == "__main__":
    main()

    from creature import Forager, THETA_LEN
    t = Forager(0, 0)
    assert (len(t.brain.fc1.weight.view(-1)) +
            len(t.brain.fc2.weight.view(-1))) == THETA_LEN - (
            len(t.brain.fc1.bias) + len(t.brain.fc2.bias))
    print("Brain slice length matches THETA_LEN – fix applied!")
