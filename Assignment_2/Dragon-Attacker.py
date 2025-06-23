from dataclasses import dataclass
from vi import Agent, Config, Simulation
import random

# --- Simulation configuration ---
@dataclass
class SimConfig(Config):
    radius: float = 30
    speed: float = 1.0
    prey_reproduction_prob: float = 0.0005
    predator_death_prob: float = 0.0025
    predator_reproduction_chance: float = 1
    duration: int = 60 * 60 * 0.5  # 30-minute simulated time

# --- Prey agent ---
class Prey(Agent):
    def update(self):
        self.pos += self.move
        self.save_data('kind', "Prey")  # log position & kind
        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

# --- Predator agent ---
class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False

    def update(self):
        self.pos += self.move
        self.save_data('kind', "Predator")  # log position & kind

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
                self.reproduce()

        if not self.has_eaten and random.random() < self.config.predator_death_prob:
            self.kill()
        self.has_eaten = False

# --- Attacker Dragon agent ---
class AttackerDragon(Agent):
    dragon_speed = 4

    def update(self):
        self.save_data('kind', "AttackerDragon")  # log position & kind

        prey = (
            self.in_proximity_accuracy()
            .without_distance()
            .filter_kind(Prey)
            .first()
        )

        if prey is not None:
            # Compute direction vector toward the prey
            direction = prey.pos - self.pos

            # Normalize the direction vector
            if direction.magnitude() != 0:
                direction = direction.normalize()

            # Move in the direction of the prey
            self.pos += direction * self.dragon_speed

            # Kill the prey if within attack range
            prey.kill()
        else:
            # Default movement if no prey found
            self.pos += self.move * self.dragon_speed

# --- Run simulation ---
result_df = (
    Simulation(config=SimConfig())
    .batch_spawn_agents(60, Prey, images=["Assignment_2/images/prey_small.png"])
    .batch_spawn_agents(20, Predator, images=["Assignment_2/images/predator_small.png"])
    .batch_spawn_agents(1, AttackerDragon, images=["Assignment_2/images/attacker_small.png"])
    .run()
    .snapshots
)

import pandas as pd

# Try converting result to DataFrame
df = pd.DataFrame(result_df)

# Print structure
print("\n--- Type of result_df:", type(result_df))
print("--- First few entries ---")
for i, entry in enumerate(result_df[:5]):
    print(f"{i}: {entry} (type: {type(entry)})")

print("\n--- DataFrame columns ---")
print(df.columns)
print(df.head())

import pandas as pd
import matplotlib.pyplot as plt

# Convert Polars DF to Pandas
df = pd.DataFrame(result_df)

# Rename the columns explicitly
df.columns = ['tick', 'id', 'x', 'y', 'image_index', 'kind']

# Group by tick and kind
grouped = df.groupby(['tick', 'kind']).size().unstack(fill_value=0)

# Prey losses
prey_count = grouped.get('Prey', pd.Series(0, index=grouped.index))
prey_loss = -prey_count.diff().fillna(0)
prey_loss[prey_loss < 0] = 0

# Get dragon active ticks
dragon_ticks = df[df["kind"] == "AttackerDragon"]["tick"].unique()
dragon_kills = prey_loss.copy()
dragon_kills[~dragon_kills.index.isin(dragon_ticks)] = 0

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot 1: Predator count
axs[0].plot(grouped.index, grouped.get("Predator", pd.Series(0, index=grouped.index)), label='Predators')
axs[0].set_title("Predator Count Over Time")
axs[0].set_ylabel("Predator Count")
axs[0].grid(True)

# Plot 2: Prey count
axs[1].plot(grouped.index, grouped.get("Prey", pd.Series(0, index=grouped.index)), label='Prey', color='green')
axs[1].set_title("Prey Count Over Time")
axs[1].set_ylabel("Prey Count")
axs[1].grid(True)

# Plot 3: Prey killed by dragon
axs[2].plot(dragon_kills.index, dragon_kills, label='Dragon Kills', color='red')
axs[2].set_title("Prey Killed by Dragon Over Time")
axs[2].set_xlabel("Time Tick")
axs[2].set_ylabel("Prey Killed")
axs[2].grid(True)

plt.tight_layout()
plt.show()