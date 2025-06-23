from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import pygame

# ------------------------------
# CONFIGURATION
# ------------------------------
@dataclass
class CastleConfig(Config):
    speed: float = 1.0
    radius: int = 25
    prey_reproduction_prob: float = 0.01
    duration: int = int(60 * 60 * 0.5)

# ------------------------------
# CASTLE ZONE
# ------------------------------
CASTLE_POSITION = Vector2(500, 500)
CASTLE_RADIUS = 120

def is_inside_castle(pos):
    return (pos - CASTLE_POSITION).length() < CASTLE_RADIUS

# ------------------------------
# CASTLE MARKER AGENT
# ------------------------------
class CastleMarker(Agent):
    def initialise_agent(self):
        self.pos = CASTLE_POSITION
        self.move = Vector2(0, 0)

    def update(self):
        self.save_data("kind", "Castle")

    def change_position(self):
        pass

# ------------------------------
# PREY AGENT
# ------------------------------
class Prey(Agent):
    def initialise_agent(self):
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def update(self):
        if not is_inside_castle(self.pos) and random.random() < 0.02:
            self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

        self.pos += self.move
        self.pos.x %= 1000
        self.pos.y %= 1000

        self.save_data('kind', "Prey")

        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

# ------------------------------
# PREDATOR AGENT (with hard bounce)
# ------------------------------
class Predator(Agent):
    def initialise_agent(self):
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def update(self):
        # Predict next position
        next_pos = self.pos + self.move

        if is_inside_castle(next_pos):
            # Rotate away slightly instead of moving in
            direction = self.pos - CASTLE_POSITION
            if direction.length() > 0:
                self.move = direction.rotate(random.uniform(-45, 45)).normalize() * self.config.speed
            next_pos = self.pos + self.move
            if is_inside_castle(next_pos):
                self.move = Vector2(0, 0)  # fully block

        else:
            if random.random() < 0.02:
                self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

        self.pos += self.move
        self.pos.x %= 1000
        self.pos.y %= 1000

        self.save_data('kind', "Predator")

        # Hunt prey (only outside castle)
        has_eaten = False
        nearby_prey = self.in_proximity_accuracy().filter_kind(Prey)
        for prey_tuple in nearby_prey:
            prey = prey_tuple[0]
            if not is_inside_castle(prey.pos):
                prey.kill()
                has_eaten = True
                if random.random() < 0.1:  # ⬅️ controlled predator reproduction
                    self.reproduce()
                break

        if not has_eaten and random.random() < 0.01:  # ⬅️ predator dies if starving
            self.kill()

# ------------------------------
# RUN SIMULATION
# ------------------------------
Simulation(config=CastleConfig()) \
    .batch_spawn_agents(1, CastleMarker, images=["images/fort.png"]) \
    .batch_spawn_agents(60, Prey, images=["images/prey_small.png"]) \
    .batch_spawn_agents(40, Predator, images=["images/predator_small.png"]) \
    .run()
