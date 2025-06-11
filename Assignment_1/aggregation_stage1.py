from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import math
import pygame

@dataclass
class AggregationConfig(Config):
    speed: float = 0.5
    radius: float = 10.0
    aggregation_zone_radius: float = 120.0
    aggregation_zone_center: tuple = (500, 500)
    Tjoin: int = 30
    Tleave: int = 30

class AggregationZone:
    def __init__(self, pos: Vector2, radius: float):
        self.pos = pos
        self.radius = radius

class ZoneAgent(Agent):
    def initialise_agent(self):
        self.pos = Vector2(self.config.aggregation_zone_center)
        
    def change_position(self):
        pass
        
    def draw(self, screen):
        # Draw a more visible green circle for the aggregation zone
        radius = int(self.config.aggregation_zone_radius)
        
        # Create a surface for the semi-transparent circle
        surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        
        # Draw a filled semi-transparent circle
        pygame.draw.circle(
            screen,
            (0, 255, 0, 30),  # Very transparent green fill
            (int(self.pos.x), int(self.pos.y)),
            radius
        )
        
        # Draw the border
        pygame.draw.circle(
            screen,
            (0, 255, 0, 255),  # Solid green border
            (int(self.pos.x), int(self.pos.y)),
            radius,
            width=2
        )

class AggregationAgent(Agent):
    WANDERING, JOIN, STILL, LEAVE = range(4)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = self.WANDERING
        self.state_timer = 0

    def initialise_agent(self):
        self.state = self.WANDERING
        self.state_timer = 0
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def change_position(self):
        # Use config values directly for the zone
        zone_center = Vector2(self.config.aggregation_zone_center)
        zone_radius = self.config.aggregation_zone_radius
        in_zone = (self.pos - zone_center).length() < zone_radius

        neighbors = self.in_proximity_accuracy()
        n = sum(1 for agent, _ in neighbors if (agent.pos - zone_center).length() < zone_radius)

        a, b = 1.70188, 3.88785
        PJoin = 0.03 + 0.48 * (1 - math.exp(-a * n)) if in_zone else 0
        PLeave = math.exp(-b * n) if in_zone else 1

        speed = self.config.speed * (0.3 if in_zone else 1.0)
        self.move = self.move.normalize() * speed if self.move.length() != 0 else Vector2(1, 0) * speed

        if self.state == self.WANDERING:
            if random.random() < 0.02:
                angle = random.uniform(-45, 45)
                self.move = self.move.rotate(angle).normalize() * speed
            if in_zone and random.random() < PJoin:
                self.state, self.state_timer = self.JOIN, 0
            self.pos += self.move

        elif self.state == self.JOIN:
            self.state_timer += 1
            if self.state_timer > self.config.Tjoin:
                self.state, self.state_timer = self.STILL, 0
            self.pos += self.move.normalize() * self.config.speed * 0.1

        elif self.state == self.STILL:
            self.move = Vector2(0, 0)
            self.state_timer += 1
            if self.state_timer > self.config.Tleave and random.random() < PLeave:
                self.state, self.state_timer = self.LEAVE, 0

        elif self.state == self.LEAVE:
            if self.move.length() < 0.01:
                angle = random.uniform(0, 360)
                self.move = Vector2(1, 0).rotate(angle).normalize() * self.config.speed
            self.state_timer += 1
            if self.state_timer > 10:
                self.state, self.state_timer = self.WANDERING, 0
            self.pos += self.move

        self._wrap_position()

    def _wrap_position(self):
        self.pos.x %= 1000
        self.pos.y %= 1000

zone = AggregationZone(Vector2(*AggregationConfig.aggregation_zone_center), AggregationConfig.aggregation_zone_radius)
AggregationAgent.zone = zone

sim = Simulation(
    AggregationConfig(
        image_rotation=True,
        speed=0.2,
        radius=10,
        fps_limit=0,
    )
)

# Spawn zone agent with a placeholder image but we'll draw our own circle
sim.batch_spawn_agents(
    1,
    ZoneAgent,
    images=["images/circle.png"]  # Need an image for the framework, but our draw method will handle visualization
)

sim.batch_spawn_agents(
    100,
    AggregationAgent,
    images=["images/triangle.png"]
).run()
