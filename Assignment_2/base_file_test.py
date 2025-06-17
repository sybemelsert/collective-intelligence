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
    prey_reproduction_prob: float = 0.0012
    predator_death_prob: float = 0.002
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

# Launch simulation
result_df = (
    HeadlessSimulation(config=SimConfig(duration=60 * 60 * 4))
    .batch_spawn_agents(60, Prey, images=["Assignment_2/images/prey_small.png"])
    .batch_spawn_agents(20, Predator, images=["Assignment_2/images/predator_small.png"])
    .run()
    .snapshots
)

# === Save CSV ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"Assignment_2/test_results/snapshot_data_{timestamp}.csv"
#result_df.write_csv(csv_path)
print(f"Saved snapshot data to {csv_path}")

# === Plot 1: Population Over Time ===
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
plt.title("Agent Population Over Time")
plt.xlabel("Frame")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plot1_path = f"Assignment_2/test_results/population_over_time_{timestamp}.png"
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Saved population plot to {plot1_path}")
