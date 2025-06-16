from dataclasses import dataclass
from vi import Agent, Config, Simulation
import random

@dataclass
class SimConfig(Config): #switch numbers here for different results
    radius: float = 10
    speed: float = 1.0
    prey_reproduction_prob: float = 0.001
    predator_death_prob: float = 0.005
    predator_reproduction_chance: float = 1

class Prey(Agent):
    def update(self):
        self.pos += self.move

        # Asexual reproduction
        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False

    def update(self):
        self.pos += self.move

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
(
    Simulation(config=SimConfig())
    .batch_spawn_agents(60, Prey, images=["Assignment_2/images/prey_small.png"])
    .batch_spawn_agents(20, Predator, images=["Assignment_2/images/predator_small.png"])
    .run()
)
