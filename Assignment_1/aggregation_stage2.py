from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import math
import pygame
import matplotlib.pyplot as plt

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
# GLOBAL TRACKING LISTS
# ------------------------------
agent_counts_inside = []
agent_counts_outside = []

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
        screen = pygame.display.get_surface()
        for zone in AggregationAgent.zones:
            pygame.draw.circle(screen, (255, 255, 0), (int(zone.pos.x), int(zone.pos.y)), int(zone.radius), width=2)

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
# CUSTOM SIMULATION CLASS
# ------------------------------
class AggregationSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tick_count = 0
        self.max_ticks = 1000  # Limit to 1000 frames
        self.running = True

    def run(self):
        while self.running:
            self.tick()
        pygame.quit()  # Close window when done

    def tick(self):
        if self.tick_count >= self.max_ticks:
            self.running = False
            return

        super().tick()

        inside_count = 0
        for agent in self._agents:
            for zone in AggregationAgent.zones:
                if (agent.pos - zone.pos).length() < zone.radius:
                    inside_count += 1
                    break
        outside_count = len(self._agents) - inside_count
        agent_counts_inside.append(inside_count)
        agent_counts_outside.append(outside_count)

        self.tick_count += 1

# ------------------------------
# RUN SIMULATION
# ------------------------------

radiusses = [80, 80] # We can change this variable to determine whether we want to have different radiuses or not
# Experiment values are: [80, 80] [100, 100], [130, 90], [140, 80]

AggregationAgent.zones = [
    AggregationZone(Vector2(225, 400), radiusses[0]),
    AggregationZone(Vector2(525, 400), radiusses[1])
]

sim = AggregationSimulation(
    AggregationConfig(
        image_rotation=True,
        speed=1,
        radius=10,
        fps_limit=30,
    )
)

sim.batch_spawn_agents(
    100,
    AggregationAgent,
    images=["Assignment_1/images/triangle.png"]
).run()

# ------------------------------
# PLOT RESULTS
# ------------------------------
plt.plot(agent_counts_inside, label='Inside Zones')
plt.plot(agent_counts_outside, label='Outside Zones')
plt.xlabel('Time Step')
plt.ylabel('Number of Agents')
plt.title('Agent Distribution Over Time')
plt.legend()
plt.show()