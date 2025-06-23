from dataclasses import dataclass
from vi import Agent, Config, Simulation
import random

# --- Simulation configuration ---
@dataclass
class SimConfig(Config):
    radius: float = 10
    speed: float = 1.0
    prey_reproduction_prob: float = 0.001
    predator_death_prob: float = 0.005
    predator_reproduction_chance: float = 1
    duration: int = 60 * 60 * 0.5  # 30-minute simulated time

# --- Prey agent ---
class Prey(Agent):
    def update(self):
        self.pos += self.move
        self.save_data('kind', "Prey")  # Log position & kind
        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

# --- Predator agent ---
class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False

    def update(self):
        self.pos += self.move
        self.save_data('kind', "Predator")

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

# --- Protector Dragon agent ---
class ProtectorDragon(Agent):
    def update(self):
        self.pos += self.move
        self.save_data('kind', "ProtectorDragon")

        # Kill nearby predators by unpacking from (agent, distance) tuples
        nearby_predators = self.in_proximity_accuracy().filter_kind(Predator)
        for predator_tuple in nearby_predators:
            predator = predator_tuple[0]
            predator.kill()

# --- Run simulation ---
result_df = (
    Simulation(config=SimConfig())
    .batch_spawn_agents(60, Prey, images=["images/prey_small.png"])
    .batch_spawn_agents(20, Predator, images=["images/predator_small.png"])
    .batch_spawn_agents(1, ProtectorDragon, images=["images/shield_small.png"])
    .run()
    .snapshots
)
