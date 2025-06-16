from dataclasses import dataclass
from vi import Agent, Config, HeadlessSimulation
import random
import polars as pl
import matplotlib.pyplot as plt
import datetime

@dataclass
class SimConfig(Config):
    radius: float = 15
    speed: float = 1.0
    prey_reproduction_prob: float = 0.001
    predator_death_prob: float = 0.002
    predator_reproduction_chance: float = 1
    duration: int = 60 * 60 * 2 # Simulation duration in minutes

class Prey(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.change_image(0)  # Explicitly set image index for prey
    
    def update(self):
        self.pos += self.move
        if random.random() < self.config.prey_reproduction_prob:
            child = self.reproduce()
            child.change_image(0)  # Ensure offspring are also prey
    
class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.change_image(1)  # Explicitly set image index for predator
        self.has_eaten = False
   
    def update(self):
        self.pos += self.move
        prey = (
            self.in_proximity_accuracy()
            .without_distance()
            .filter_kind(Prey)
            .first()
        )

        if prey is not None:
            prey.kill()
            self.has_eaten = True
            if random.random() < self.config.predator_reproduction_chance:
                child = self.reproduce()
                child.change_image(1)  # Ensure offspring are also predators

        if not self.has_eaten and random.random() < self.config.predator_death_prob:
            self.kill()
        self.has_eaten = False

# Run simulation
for run in range(1, 2):  # Change range for more runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"Assignment_2/test_results/prey_predator_plot_{timestamp}_run{run}.png"
    data_filename = f"Assignment_2/test_results/prey_predator_data_{timestamp}_run{run}.csv"

    # Create simulation with both images
    sim = (
        HeadlessSimulation(SimConfig())
        .batch_spawn_agents(60, Prey, images=[
            "Assignment_2/images/prey_small.png",  # image_index 0
            "Assignment_2/images/predator_small.png"  # image_index 1
        ])
        .batch_spawn_agents(20, Predator, images=[
            "Assignment_2/images/prey_small.png",  # image_index 0
            "Assignment_2/images/predator_small.png"  # image_index 1
        ])
    )

    # Run and process data
    df = (
        sim.run()
        .snapshots
        .group_by(["frame", "image_index"])
        .agg(pl.count("id").alias("count"))
        .sort("frame")
    )

    # Add smoothing
    window_size = 500
    df = df.with_columns(
        pl.col("count")
        .rolling_mean(window_size, min_periods=1)
        .over("image_index")
        .alias("count_smoothed")
    )

    # Save data
    df.write_csv(data_filename)

    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Map image indices to labels
    image_labels = {0: "Prey", 1: "Predator"}
    
    for image_index in [0, 1]:
        sub_df = df.filter(pl.col("image_index") == image_index)
        plt.plot(sub_df["frame"], sub_df["count_smoothed"], label=image_labels[image_index])

    plt.xlabel("Frame")
    plt.ylabel("Number of Agents (Smoothed)")
    plt.title("Prey vs Predator Population Dynamics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    print(f"Saved plot to {plot_filename} and data to {data_filename}")