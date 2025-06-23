from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import pygame
import math

@dataclass
class SimConfig(Config):
    radius: float = 10
    speed: float = 1.0
    prey_reproduction_prob: float = 0.001
    predator_death_prob: float = 0.0025
    predator_reproduction_chance: float = 1
    castle_capacity: int = 10
    max_castle_stay: int = 60 * 3   # 5 seconds if 60 FPS
    castle_radius: float = 50
    repel_strength: float = 1.5
    detection_radius: float = 100
    eating_radius: float = 15
    castle_wander_speed: float = 0.3

class Castle(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preys_in_castle = {}
        self.config.radius = self.config.castle_radius

    def update(self):
        self.save_data('kind', 'Castle')
        screen = pygame.display.get_surface()
        if screen is not None:
            pygame.draw.circle(screen, (0, 0, 255), (int(self.pos.x), int(self.pos.y)), 
                             int(self.config.castle_radius), width=1)

    def change_position(self):
        pass

    def allow_entry(self, prey):
        return len(self.preys_in_castle) < self.config.castle_capacity

    def enter(self, prey):
        self.preys_in_castle[prey] = 0
        # Random position within castle radius when entering (but not too close to center)
        angle = random.uniform(0, 2*math.pi)
        min_distance = prey.config.radius * 2  # Minimum distance from center
        max_distance = self.config.castle_radius - prey.config.radius
        distance = random.uniform(min_distance, max_distance)
        prey.pos = self.pos + Vector2(math.cos(angle), math.sin(angle)) * distance

    def tick(self):
        to_remove = []
        for prey, ticks in self.preys_in_castle.items():
            if not prey.alive or ticks >= self.config.max_castle_stay:
                to_remove.append(prey)
            else:
                self.preys_in_castle[prey] += 1
        for prey in to_remove:
            self.preys_in_castle.pop(prey, None)

class Prey(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_castle = False
        self.castle_timer = 0
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed
        self.current_castle = None
        self.wander_timer = 0

    def update(self):
        if self.in_castle:
            self.save_data('kind', 'Prey (in castle)')
            self.castle_timer += 1
            
            # Wandering behavior inside castle
            self.wander_timer += 1
            if self.wander_timer > 30:  # Change direction periodically
                self.wander_timer = 0
                angle = random.uniform(-45, 45)
                self.move = self.move.rotate(angle).normalize() * self.config.castle_wander_speed
            
            # Calculate new position
            new_pos = self.pos + self.move
            
            # Verify new position stays within castle bounds
            if self.current_castle:
                to_center = new_pos - self.current_castle.pos
                distance = to_center.length()
                max_distance = self.current_castle.config.castle_radius - self.config.radius
                
                if distance <= max_distance:
                    # Move freely if within bounds
                    self.pos = new_pos
                else:
                    # Adjust position to stay exactly at the boundary if trying to move out
                    self.pos = self.current_castle.pos + (to_center.normalize() * max_distance)
                    # Bounce off the wall
                    normal = to_center.normalize()
                    self.move = self.move.reflect(normal) * 0.8
            
            if self.castle_timer >= self.config.max_castle_stay:
                self.leave_castle()
            return
        else:
            self.save_data('kind', 'Prey')
        
        # Castle detection
        for agent, dist in self.in_proximity_accuracy().filter_kind(Castle):
            if dist < self.config.detection_radius:
                castle = agent
                if (self.pos - castle.pos).length() <= castle.config.castle_radius and castle.allow_entry(self):
                    self.enter_castle(castle)
                    return
                
                repel = (self.pos - castle.pos)
                if repel.length() > 0:
                    repel = repel.normalize()
                self.move += repel * self.config.repel_strength

        # Normal movement
        self.move = self.move.normalize() * self.config.speed
        self.pos += self.move

        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

    def enter_castle(self, castle):
        self.in_castle = True
        self.castle_timer = 0
        self.current_castle = castle
        castle.enter(self)
        # Start with random movement direction
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.castle_wander_speed

    def leave_castle(self):
        if self.current_castle:
            repel_vector = (self.pos - self.current_castle.pos)
            if repel_vector.length() > 0:
                repel_vector = repel_vector.normalize()
            else:
                repel_vector = Vector2(1, 0)
            self.pos = self.current_castle.pos + repel_vector * (self.config.castle_radius + self.config.radius)
            self.move = repel_vector * self.config.speed * 1.5
        self.in_castle = False
        self.castle_timer = 0
        self.current_castle = None
        self.wander_timer = 0

class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def update(self):
        self.save_data('kind', 'Predator')

        # Castle avoidance
        for agent, dist in self.in_proximity_accuracy().filter_kind(Castle):
            if dist < self.config.detection_radius:
                castle = agent
                repel = (self.pos - castle.pos)
                if repel.length() > 0:
                    repel = repel.normalize()
                self.move += repel * self.config.repel_strength * 2.0

        # Normal movement
        self.move = self.move.normalize() * self.config.speed
        self.pos += self.move

        # Hunting
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

# Run simulation
result_df = (
    Simulation(config=SimConfig(duration=60 * 60 * 0.5))
    .spawn_agent(Castle, images=["Assignment_2/images/fort.png"])
    .batch_spawn_agents(20, Prey, images=["Assignment_2/images/prey_small.png"])
    .batch_spawn_agents(10, Predator, images=["Assignment_2/images/predator_small.png"])
    .run()
    .snapshots
)

print(result_df)