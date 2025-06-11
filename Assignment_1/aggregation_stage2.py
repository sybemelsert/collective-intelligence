from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import math
import pygame

# ------------------------------
# CONFIGURATION
# ------------------------------
@dataclass
class AggregationConfig(Config):
    speed: float = 0.5
    radius: float = 10.0
    aggregation_zone_radius: float = 120.0
    Tjoin: int = 30
    Tleave: int = 30

# ------------------------------
# AGGREGATION ZONE
# ------------------------------
class AggregationZone:
    def __init__(self, pos: Vector2, radius: float):
        self.pos = pos
        self.radius = radius

# ------------------------------
# AGENT DEFINITION
# ------------------------------
class AggregationAgent(Agent):
    WANDERING, JOIN, STILL, LEAVE = range(4)
    zones = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = self.WANDERING
        self.state_timer = 0

    def initialise_agent(self):
        self.state = self.WANDERING
        self.state_timer = 0
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def update(self):
        # Draw aggregation zone (called every frame)
        screen = pygame.display.get_surface()
        for zone in AggregationAgent.zones:
            pos = zone.pos
            radius = zone.radius
            pygame.draw.circle(screen, (255, 255, 0), (int(pos.x), int(pos.y)), int(radius), width=2)

    def change_position(self):
        neighbors = self.in_proximity_accuracy()
        in_zone = False
        n = 0

        for zone in self.zones:
            if (self.pos - zone.pos).length() < zone.radius:
                in_zone = True
                n += sum(1 for agent, _ in neighbors if (agent.pos - zone.pos).length() < zone.radius)

        # Probability formulas
        a, b = 1.70188, 3.88785
        PJoin = 0.03 + 0.48 * (1 - math.exp(-a * n)) if in_zone else 0
        PLeave = math.exp(-b * n) if in_zone else 1

        # State transitions
        if self.state == self.WANDERING:
            if random.random() < 0.02:
                angle = random.uniform(-45, 45)
                self.move = self.move.rotate(angle)
                if self.move.length() == 0:
                    self.move = Vector2(1, 0)
                self.move = self.move.normalize() * self.config.speed
            if in_zone and random.random() < PJoin:
                self.state, self.state_timer = self.JOIN, 0
            self.pos += self.move

        elif self.state == self.JOIN:
            self.state_timer += 1
            if self.state_timer > self.config.Tjoin:
                self.state, self.state_timer = self.STILL, 0
            self.pos += self.move.normalize() * self.config.speed * 0.2

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

# ------------------------------
# RUN SIMULATION
# ------------------------------

similar_radius = False # We can change this variable to determine whether we want to have different radiuses or not

if similar_radius == True:
    AggregationAgent.zones = [
    AggregationZone(Vector2(225, 400), 100),
    AggregationZone(Vector2(525, 400), 100)
    ]

else:
    AggregationAgent.zones = [
    AggregationZone(Vector2(225, 400), 140),
    AggregationZone(Vector2(525, 400), 80)
    ]

sim = Simulation(
    AggregationConfig(
        image_rotation=True,
        speed=0.2,
        radius=10,
        fps_limit=0,
    )
)

sim.batch_spawn_agents(
    100,
    AggregationAgent,
    images=["Assignment_1/images/triangle.png"]
).run()
