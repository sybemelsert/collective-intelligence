# Defender-only simulation with static barn (Castle) and custom defender speed
from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import math

@dataclass
class SimConfig(Config):
    radius: float = 30
    speed: float = 1.0
    prey_reproduction_prob: float = 0.0005
    predator_death_prob: float = 0.0025
    predator_reproduction_chance: float = 1
    castle_capacity: int = 10
    max_castle_stay: int = 60 * 3
    castle_radius: float = 50
    repel_strength: float = 1.5
    detection_radius: float = 100
    eating_radius: float = 15
    duration: int = 60 * 60 * 0.5

class Castle(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preys_in_castle = {}
        self.config.radius = self.config.castle_radius
        self.move = Vector2(0, 0)

    def update(self):
        self.save_data('kind', 'Castle')

    def change_position(self):
        pass

    def allow_entry(self, prey):
        return len(self.preys_in_castle) < self.config.castle_capacity

    def enter(self, prey):
        self.preys_in_castle[prey] = 0
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(prey.config.radius, self.config.castle_radius - (prey.config.radius / 0.7))
        prey.pos = self.pos + Vector2(math.cos(angle), math.sin(angle)) * distance
        prey.move = Vector2(0, 0)

    def leave(self, prey):
        self.preys_in_castle.pop(prey, None)

    def tick(self):
        to_remove = []
        for prey, ticks in self.preys_in_castle.items():
            if not prey.alive or ticks >= self.config.max_castle_stay:
                to_remove.append(prey)
            else:
                self.preys_in_castle[prey] += 1
        for prey in to_remove:
            self.leave(prey)

class Prey(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_castle = False
        self.castle_timer = 0
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed
        self.current_castle = None

    def update(self):
        if self.in_castle:
            self.save_data('kind', 'Prey')
            self.castle_timer += 1
            if self.castle_timer >= self.config.max_castle_stay:
                self.leave_castle()
            return

        self.save_data('kind', 'Prey')

        for agent, dist in self.in_proximity_accuracy().filter_kind(Castle):
            if dist < self.config.detection_radius:
                if (self.pos - agent.pos).length() <= agent.config.castle_radius and agent.allow_entry(self):
                    self.enter_castle(agent)
                    return
                repel = (self.pos - agent.pos)
                if repel.length() > 0:
                    repel = repel.normalize()
                    self.move += repel * self.config.repel_strength

        self.move = self.move.normalize() * self.config.speed
        self.pos += self.move

        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

    def enter_castle(self, castle):
        self.in_castle = True
        self.castle_timer = 0
        self.current_castle = castle
        castle.enter(self)

    def leave_castle(self):
        if self.current_castle:
            self.current_castle.leave(self)
            repel_vector = (self.pos - self.current_castle.pos).normalize()
            self.pos = self.current_castle.pos + repel_vector * (self.config.castle_radius + self.config.radius)
            self.move = repel_vector * self.config.speed * 1.5
        self.in_castle = False
        self.castle_timer = 0
        self.current_castle = None

class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def update(self):
        self.save_data('kind', 'Predator')

        for agent, dist in self.in_proximity_accuracy().filter_kind(Castle):
            if dist < self.config.detection_radius:
                repel = (self.pos - agent.pos).normalize()
                self.move += repel * self.config.repel_strength * 2.0

        self.move = self.move.normalize() * self.config.speed
        self.pos += self.move

        for prey, dist in self.in_proximity_accuracy().filter_kind(Prey):
            if not prey.in_castle and dist < self.config.eating_radius:
                prey.kill()
                self.has_eaten = True
                if random.random() < self.config.predator_reproduction_chance:
                    self.reproduce()
                break

        if not self.has_eaten and random.random() < self.config.predator_death_prob:
            self.kill()
        self.has_eaten = False

class ProtectorDragon(Agent):
    speed = 1 # Custom speed for the defender

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.speed

    def update(self):
        self.save_data('kind', 'ProtectorDragon')
        self.move = self.move.normalize() * self.speed
        self.pos += self.move
        for predator_tuple in self.in_proximity_accuracy().filter_kind(Predator):
            predator = predator_tuple[0]
            predator.kill()

# Run simulation
result_df = (
    Simulation(config=SimConfig())
    .spawn_agent(Castle, images=["Assignment_2/images/barn.png"])
    .batch_spawn_agents(60, Prey, images=["Assignment_2/images/prey_small.png"])
    .batch_spawn_agents(20, Predator, images=["Assignment_2/images/predator_small.png"])
    .batch_spawn_agents(1, ProtectorDragon, images=["Assignment_2/images/shield_small.png"])
    .run()
    .snapshots
)

print(result_df)