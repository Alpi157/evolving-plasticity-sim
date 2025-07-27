import os,csv,random,numpy as np,config
from creature import Forager,THETA_LEN
from evolve import run_generation
from world import World
from visualize import run_demo
def main():
    profile=input("Shift profile (rare/frequent/normal): ").strip().lower()
    if profile=="rare": sh_min,sh_max=15,20
    elif profile=="frequent": sh_min,sh_max=1,3
    else: sh_min,sh_max=None,None
    chk_raw=input("Enter generation checkpoints (e.g. 1,5,10): ")
    checkpoints=sorted({int(x) for x in chk_raw.replace(" ","").split(",") if x.isdigit()})
    os.makedirs(config.LOG_DIR,exist_ok=True)
    os.makedirs(config.PLOTS_DIR,exist_ok=True)
    os.makedirs(config.GENOME_DIR,exist_ok=True)
    population=[Forager(random.randrange(config.WORLD_WIDTH),random.randrange(config.WORLD_HEIGHT)) for _ in range(config.POPULATION_SIZE)]
    world=World(shift_min=sh_min,shift_max=sh_max)
    with open(os.path.join(config.LOG_DIR,"stats_log.csv"),"w",newline="") as fcsv,open(os.path.join(config.LOG_DIR,"postshift_log.csv"),"w",newline="") as fshift:
        stats_writer=csv.DictWriter(fcsv,fieldnames=["generation","best_fitness","avg_fitness","median_eta","median_lambda"])
        shift_writer=csv.DictWriter(fshift,fieldnames=["shift_id","generation","best_fitness","avg_fitness"])
        stats_writer.writeheader();shift_writer.writeheader()
        shift_mode=False;postshift_counter=0;shift_id=0
        for gen in range(1,config.NUM_GENERATIONS+1):
            prev_state=world.generations_since_shift
            world.step_generation()
            did_shift=(world.generations_since_shift==0 and prev_state!=0)
            population,stats=run_generation(population,world)
            stats_writer.writerow({"generation":gen,"best_fitness":stats["best_fitness"],"avg_fitness":stats["avg_fitness"],"median_eta":stats["median_eta"],"median_lambda":stats["median_lambda"]})
            print(f"Gen {gen:4d} | best {stats['best_fitness']:.2f} | avg {stats['avg_fitness']:.2f} | η̃ {stats['median_eta']:.4f} | λ̃ {stats['median_lambda']:.4f}")
            if did_shift:
                print(f"[ENV SHIFT] at Gen {gen}")
                shift_mode=True;postshift_counter=0;shift_id+=1
            if shift_mode:
                shift_writer.writerow({"shift_id":shift_id,"generation":gen,"best_fitness":stats["best_fitness"],"avg_fitness":stats["avg_fitness"]})
                postshift_counter+=1
                if postshift_counter>=10: shift_mode=False
            if gen in checkpoints:
                input(f"Watch Generation {gen} ...")
                run_demo(population,world)
            if gen%config.SAVE_INTERVAL==0 or gen==config.NUM_GENERATIONS:
                for i,ag in enumerate(population):
                    np.save(os.path.join(config.GENOME_DIR,f"gen{gen:04d}_ind{i:03d}.npy"),ag.genome)
    print("Done. Logs in",config.LOG_DIR)
if __name__=="__main__":
    main()
