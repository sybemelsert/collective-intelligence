from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
import random
import polars as pl
import matplotlib.pyplot as plt
import datetime

@dataclass
class SimConfig(Config): #switch numbers here for different results
    radius: float = 15
    speed: float = 1.0
    prey_reproduction_prob: float = 0.001
    predator_death_prob: float = 0.001
    predator_reproduction_chance: float = 1

class Prey(Agent):
    def update(self):
        self.pos += self.move
        self.save_data('kind', "Prey")

        # Asexual reproduction
        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False

    def update(self):
        self.pos += self.move
        self.save_data('kind', "Predator")

        # Look for prey nearby
        prey = (
            self.in_proximity_accuracy()
            .without_distance()
            .filter_kind(Prey)
            .first()
        )

        # If prey is found, eat and possibly reproduce
        if prey is not None:
            prey.kill()
            self.has_eaten = True

            # ✅ Predator, not prey, reproduces
            if random.random() < self.config.predator_reproduction_chance:
                self.reproduce()

        # Spontaneous death if didn't eat
        if not self.has_eaten and random.random() < self.config.predator_death_prob:
            self.kill()

        self.has_eaten = False


num_runs = 3  # Number of simulation runs
for i in range(num_runs):
    print(f"\n=== Running simulation {i + 1} of {num_runs} ===")

    result_df = (
        HeadlessSimulation(config=SimConfig(duration=60 * 60 * 2)) #2 minutes simulation
        .batch_spawn_agents(60, Prey, images=["Assignment_2/images/prey_small.png"])
        .batch_spawn_agents(20, Predator, images=["Assignment_2/images/predator_small.png"])
        .run()
        .snapshots
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{i+1}_{timestamp}"
    csv_path = f"Assignment_2/test_results/snapshot_data_{run_id}.csv"
    plot1_path = f"Assignment_2/test_results/population_over_time_{run_id}.png"

    # result_df.write_csv(csv_path)
    print(f"✅ Saved snapshot data to {csv_path}")

    df_grouped = (
        result_df
        .group_by(["frame", "kind"])
        .agg(pl.count("id").alias("count"))
        .pivot(index="frame", values="count", on="kind")
        .fill_null(0)
        .sort("frame")
    )

    plt.figure(figsize=(10, 6))
    for kind in df_grouped.columns[1:]:
        plt.plot(df_grouped["frame"], df_grouped[kind], label=kind)
    plt.title(f"Population Over Time (Run {i+1})")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"✅ Saved population plot to {plot1_path}")

    time.sleep(1)  # prevent identical timestamps